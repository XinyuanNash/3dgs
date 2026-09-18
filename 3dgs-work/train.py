#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel
from scene.strategy import get_strategy, OPTIONAL_MODULES
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

# Cross-cutting train-time modules (P0-Bilateral-v2 / P2-PPISP / P2-ADMM /
# P1-5.7 scale_reg) are imported via `from scene.strategy import OPTIONAL_MODULES`
# above; the side-effect of importing scene/strategy/__init__.py registers each
# wrapper into OPTIONAL_MODULES. No per-module try/except needed here —
# each wrapper handles its own import failure (submodule build, etc.) at
# __init__ time and emits a clean error message.

def _build_optional_modules(opt, scene):
    """Instantiate every @register-decorated OptInTrainModule subclass.

    Passes ``scene`` only to modules whose ``requires_scene`` returns True
    (BilateralGrid + PPISP read n_cams from Scene). Returns the live
    module list, in registration order.

    Mirrors LFS trainer.cpp's "TrainableModule construction" pattern.
    """
    modules = []
    for cls in OPTIONAL_MODULES:
        try:
            if cls.__dict__.get("requires_scene", True):
                mod = cls(opt, scene)
            else:
                mod = cls(opt)
        except Exception as e:
            print(f"[OptInTrainModule] failed to construct {cls.__name__}: {e}")
            sys.exit(1)
        modules.append(mod)
    return modules


def _opt_in_apply_chain(modules, image, vind, iteration=None):
    """Image-space transform chain (BilateralGrid → PPISP).

    Each module's ``apply`` is a CHW→CHW function. When frozen or opted
    out, the module returns the input unchanged — so the chain is
    unconditionally safe to call.

    ``iteration`` is threaded through so freeze-gated modules (PPISP)
    can skip the forward pass when not engaged. BilateralGrid's CUDA
    kernel ignores it (frozen = identity either way).
    """
    for mod in modules:
        image = mod.apply(image, vind=vind, iteration=iteration)
    return image


def _opt_in_loss_terms(modules):
    """Sum extra loss terms from each enabled module (scale_reg + PPISP reg).

    Returns 0.0 when no module contributes; otherwise the sum of
    ``module.reg_loss()`` — each term lives on cuda so autograd tracks it.
    """
    loss_extra = torch.tensor(0.0, device="cuda")
    for mod in modules:
        loss_extra = loss_extra + mod.reg_loss()
    return loss_extra


def _opt_in_step_chain(modules, iteration, gaussians=None):
    """Per-iter universal loop: step (gated by engaged) + zero_grad (always) + scheduler_step (always).

    `gaussians` is threaded through kwargs for ADMMModule (which passes it
    to ADMMController.step); other modules ignore it.
    """
    for mod in modules:
        if mod.engaged(iteration):
            mod.step(iteration, gaussians=gaussians)
        mod.zero_grad()
        mod.scheduler_step()


def _opt_in_eval_chain(modules, image, vind):
    """Re-apply image transforms in training_report (eval path).

    Caller wraps in ``torch.no_grad()`` so no autograd graph is built.
    """
    for mod in modules:
        image = mod.eval_apply(image, vind=vind)
    return image


def _opt_in_finalize(modules, gaussians, scene, opt, strategy,
                     saving_iterations, checkpoint_iterations, iteration):
    """One-shot post-loop hook (ADMMModule re-saves PLY/chkpnt after prune)."""
    for mod in modules:
        mod.finalize(gaussians, scene, opt, strategy,
                     saving_iterations, checkpoint_iterations, iteration)


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from,
             optional_modules=None):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    # Cross-cutting module availability checks (--use_bilateral_grid, --use_ppisp)
    # are handled inside each wrapper's __init__ — a clean ImportError
    # propagates if the user opted in but the underlying dep is missing.

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    # P1-7+ (OPTIMIZATIONS.md §13.6): lazily create the rasterizer's persistent
    # per-Gaussian pixel-error buffer. Only meaningful when MCMC is the active
    # strategy (--densification_strategy mcmc, or --max_cap > 0 auto-promotion);
    # the buffer is otherwise an idle allocation. The handle is owned by the
    # C++ RasterizerState and freed in the finally block below.
    if opt.p1_7_plus:
        handle = gaussians.ensure_error_state()
        print(f"[P1-7+] allocated rasterizer error_state_handle={handle} "
              f"(p1_7_plus=True, strategy='{opt.densification_strategy}')")
    else:
        handle = 0
    # P1-MultiStrategy (OPTIMIZATIONS.md §13.7): auto-select MCMC when the
    # caller passes --max_cap > 0 but didn't explicitly request another
    # strategy. Mirrors LFS convention where --max-cap >= 0 implies MCMC.
    if opt.max_cap > 0 and opt.densification_strategy == "default":
        opt.densification_strategy = "mcmc"
        print(f"[P1] auto-select: --max_cap={opt.max_cap} > 0 -> strategy='mcmc'")
    # P1-MultiStrategy: dispatch to the selected strategy's post_backward hook.
    # Strategy is instantiated AFTER training_setup so the optimizer exists.
    # It captures a reference to (model, opt, dataset) and is otherwise stateless.
    strategy = get_strategy(opt.densification_strategy)(gaussians, opt, dataset)

    # P0-Bilateral-v2 / P2-PPISP / P2-ADMM / P1-5.7 scale_reg are all
    # opt-in cross-cutting train-time modules; see scene/strategy/*.py.
    # Each @register-decorated OptInTrainModule subclass is constructed
    # once via _build_optional_modules(); train.py never inspects module
    # types directly — it dispatches via the abstract interface.
    if optional_modules is None:
        optional_modules = []
    # Modules whose construction needs Scene (n_cams etc.) are passed scene;
    # others (scale_reg, admm) are constructed with just opt. The helper
    # routes this based on each module's `requires_scene` property.
    modules = _build_optional_modules(opt, scene)

    if checkpoint:
        (model_params, saved_strategy_name, saved_strategy_sd, first_iter) = torch.load(checkpoint)
        # P1-MultiStrategy (OPTIMIZATIONS.md §13.7): refuse to silently
        # load a checkpoint trained with a different strategy — the
        # optimizer state + Gaussian buffers are not portable across
        # strategies (MCMC maintains opacity_visible_count, etc.).
        if saved_strategy_name != strategy.name:
            raise RuntimeError(
                f"strategy mismatch: checkpoint was trained with "
                f"{saved_strategy_name!r}, but current run uses "
                f"{strategy.name!r}. Refusing to restore — re-train from "
                f"scratch or pass --densification_strategy "
                f"{saved_strategy_name}."
            )
        gaussians.restore(model_params, opt)
        strategy.load_state_dict(saved_strategy_sd)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=False)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        # P1-7+ (OPTIMIZATIONS.md §13.6): need gt_image for the rasterizer's
        # pixel-error kernel BEFORE the render() call (the C++ wrapper
        # validates shape + pointer at call time, not lazily). Pull it
        # up here; the legacy loss path also uses this variable below.
        gt_image = viewpoint_cam.original_image.cuda()

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=False,
                            compute_error=bool(opt.p1_7_plus),
                            gt_image=gt_image if opt.p1_7_plus else None,
                            error_state_handle=handle if opt.p1_7_plus else 0)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Cross-cutting image transform chain (BilateralGrid → PPISP). Each
        # module is a CHW→CHW function; frozen / opted-out modules return
        # the input unchanged, so the chain is unconditionally safe.
        # iteration is threaded so PPISP can skip the 4-stage forward when
        # frozen (avoids gradient-chain collapse when combined with
        # --igs_plus_scale_reg_weight, observed on ABAB 2026-08-27).
        image_for_loss = _opt_in_apply_chain(modules, image, vind, iteration)

        # Loss
        Ll1 = l1_loss(image_for_loss, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image_for_loss.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image_for_loss, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        # Cross-cutting regularizers (P1-5.7 scale_reg + P2-PPISP reg terms).
        # Each module returns 0 when opted out, so the chain is unconditionally
        # safe. The terms live on cuda so autograd tracks them.
        loss = loss + _opt_in_loss_terms(modules)

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp, optional_modules=modules)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification (delegated to the active Strategy — see scene/strategy/*)
            strategy.post_backward(
                iteration,
                viewspace_point_tensor,
                visibility_filter,
                radii,
                scene.cameras_extent,
                # P1-7+ (OPTIMIZATIONS.md §13.6): thread the per-Gaussian
                # pixel-error buffer into MCMC's clone/relocate scoring.
                # Empty Tensor when --p1_7_plus is off; MCMC ignores it.
                error_buffer=render_pkg.get("error_buffer"),
            )

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)
                # Strategy step hook (no-op for DefaultStrategy; MCMC injects
                # noise on _xyz here in P1-3; IGS+ may schedule edge-score
                # pre-passes in P1-5).
                strategy.step(iteration)
                # Cross-cutting per-iter step (BilateralGrid + PPISP Adam
                # step, ADMM controller step, etc). MUST run AFTER
                # loss.backward() — BilateralGrid/PPISP Adam.step() consumes
                # .grad fields populated by this iter's backward, and an
                # in-place param mutation before backward would corrupt the
                # autograd graph (RuntimeError: variable modified by inplace
                # op, observed at iter ~26K when PPISP first engaged after
                # the freeze-during-densification window). The ADMM step
                # does not consume .grad (it mutates opacity), so its
                # placement here is timing-equivalent to the pre-backward
                # call in the original train.py.
                _opt_in_step_chain(modules, iteration, gaussians=gaussians)
                # P3-Morton (doc/specs/P3-Morton-spec.md §2.4): periodic
                # Morton-order permutation of all Gaussian state during
                # the densify window. Default OFF; opt-in via
                # ``--use_morton_order``. Pure data layout — does not
                # consume .grad and does not mutate param values, so
                # placement is timing-equivalent to ADMM (per
                # [[train-py-opt-in-step-chain-post-backward]]). Trigger
                # is gated by:
                #   1. opt.use_morton_order  (master switch)
                #   2. iteration % interval == 0 (amortize cost)
                #   3. densify_from_iter <= iter <= densify_until_iter
                #      (skip before densify starts — too few Gaussians;
                #      skip after densify ends — cost > benefit, spatial
                #      locality is already settled)
                #   4. N >= opt.morton_min_gaussians  (skip small sets)
                if (opt.use_morton_order
                    and opt.densify_from_iter <= iteration <= opt.densify_until_iter
                    and (iteration % opt.morton_reorder_interval) == 0
                    and gaussians.get_xyz.shape[0] >= opt.morton_min_gaussians):
                    # Apply the Morton permutation to BOTH the Gaussian
                    # tensors (xyz/scaling/rotation/opacity/SH/Adam) AND
                    # the active strategy's per-Gaussian state
                    # (IGS+ _free_mask/_error_score_max/_edge_score_cache,
                    # MCMC _error_score_max). DefaultStrategy has no
                    # state and the no-op apply_permutation runs in <1µs.
                    # Empirically, skipping the strategy permute caused
                    # IGS+ sampling scores to drift off-slot and PSNR
                    # collapsed to ~10 dB at iter 4000 (vs ~18 baseline);
                    # see STATUS_P3-Morton.md §3.1.
                    gaussians.reorder_morton(strategy=strategy)
                # Cross-cutting step already ran BEFORE backward() (see above).
                # ADMM must run pre-backward to mutate opacity for next
                # iter's gradient; BilateralGrid + PPISP Adam steps are
                # independent of the Gaussian optimizer step ordering.
                pass

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                # P1-MultiStrategy (OPTIMIZATIONS.md §13.7): persist the
                # strategy name + state_dict alongside the model so a
                # future --start_checkpoint can refuse to load a
                # mismatched strategy. See gaussians.restore() above.
                torch.save(
                    (
                        gaussians.capture(),
                        strategy.name,
                        strategy.state_dict(),
                        iteration,
                    ),
                    scene.model_path + "/chkpnt" + str(iteration) + ".pth",
                )

    # Cross-cutting post-loop finalize (P2-ADMM: materialize_prune +
    # re-save PLY/chkpnt to reflect pruned Gaussian set). The in-loop
    # ``scene.save()`` / ``torch.save(...)`` at opt.iterations captured the
    # pre-prune state; finalize() rewrites those artefacts.
    _opt_in_finalize(modules, gaussians, scene, opt, strategy,
                     saving_iterations, checkpoint_iterations, opt.iterations)

    # P1-7+ (OPTIMIZATIONS.md §13.6): release the persistent error buffer
    # when training finishes. Idempotent. Placed at the end of training()
    # (not training_report) because `opt` is in scope here.
    if opt.p1_7_plus:
        gaussians.free_error_state()

def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp, optional_modules=None):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    # Cross-cutting image transform re-apply (BilateralGrid + PPISP) so PSNR
                    # is computed against the post-transform render (matches
                    # the training loss path). Caller is inside
                    # ``torch.no_grad()`` already.
                    image = _opt_in_eval_chain(optional_modules or [], image, viewpoint.uid)
                    image = torch.clamp(image, 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    # Cross-cutting opt-in train-time modules (BilateralGrid, PPISP, ADMM,
    # scale_reg) are constructed inside training() so they can read n_cams
    # from Scene — pass an empty list as a placeholder; the helper inside
    # training() instantiates the registered modules lazily.
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from,
             optional_modules=[])

    # All done
    print("\nTraining complete.")

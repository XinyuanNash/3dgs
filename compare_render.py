"""Compare raw rasterization vs PPISP+BilateralGrid corrected.

Goal: determine if the 'dark' rendered output is from:
- Raw Gaussians learned at wrong exposure (raw also dark), OR
- PPISP/BilateralGrid over-correcting (raw bright, corrected dark)

We do this by rendering with and without the image-space chain applied.
"""
import sys, os
sys.path.insert(0, "/home/3dgs-work")
import torch, numpy as np
from PIL import Image
from argparse import ArgumentParser

# Disable viewer + early-init PPISP/BilateralGrid
import arguments
from arguments import ModelParams, PipelineParams
from scene import Scene
from gaussian_renderer import render, GaussianModel
from scene.strategy.optional_module import OPTIONAL_MODULES
import utils.ppisp as ppisp_mod
from diff_gaussian_rasterization import BilateralGrid

JOB = "/home/jobs/20260911093928-f738e7bc"
ITER = 30000

with torch.no_grad():
    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=ITER, type=int)
    args = model.extract(parser.parse_args(f"-s {JOB} -m {JOB}/model".split()))
    args.iteration = ITER
    args.depths = ""
    args.resolution = -1
    args.data_device = "cuda"
    args.sh_degree = 3
    args.eval = True

    g = GaussianModel(args.sh_degree)
    scene = Scene(model.extract(args), g, load_iteration=args.iteration, shuffle=False)
    n_cams = len(scene.getTrainCameras()) + len(scene.getTestCameras())
    print(f"[scene] {len(scene.getTrainCameras())} train + {len(scene.getTestCameras())} test, {g.get_xyz.shape[0]} gaussians", flush=True)

    bg = torch.zeros(3, device="cuda")
    pipe = pipeline.extract(args)
    defaults = {"debug": False, "antialiasing": False, "compute_cov3D_python": False, "convert_SHs_python": False, "depth_threshold": 0.0}
    for k, v in defaults.items():
        if not hasattr(pipe, k):
            setattr(pipe, k, v)

    # Initialize FRESH PPISP + BilateralGrid (since they're not persisted)
    ppisp = ppisp_mod.PPISP(
        num_frames=n_cams, num_cameras=n_cams,
        total_iterations=args.iterations, start_iter=26000,
        lr=args.ppisp_lr,
        reg_weight_exposure_mean=args.ppisp_reg_exposure_mean,
        reg_weight_vig_center=args.ppisp_reg_vig_center,
        reg_weight_vig_channel=args.ppisp_reg_vig_channel,
        reg_weight_vig_non_pos=args.ppisp_reg_vig_non_pos,
        reg_weight_color_mean=args.ppisp_reg_color_mean,
        reg_weight_crf_channel=args.ppisp_reg_crf_channel,
    ).cuda()
    bg_grid = BilateralGrid(
        num_images=n_cams,
        grid_W=args.bilateral_grid_X,
        grid_H=args.bilateral_grid_Y,
        grid_L=args.bilateral_grid_W,
        lr=args.bilateral_grid_lr,
        total_iterations=args.iterations,
        tv_weight=args.tv_loss_weight,
        start_iter=args.bilateral_grid_start_iter,
    )
    print(f"[init] PPISP + BilateralGrid initialized fresh (params at identity)", flush=True)
    print(f"[init] PPISP exposure raw: shape={tuple(ppisp.exposure.raw.shape)} values={ppisp.exposure.raw.flatten()[:5].cpu().tolist()}", flush=True)

    test_cams = scene.getTestCameras()
    vc = test_cams[0]
    print(f"[render] cam '{vc.image_name}' shape={tuple(vc.original_image.shape)}", flush=True)

    # 1) Raw render (no PPISP/BilateralGrid)
    out_raw = render(vc, g, pipe, bg, use_trained_exp=False, separate_sh=False)
    img_raw = out_raw["render"]
    print(f"\n=== RAW (no PPISP/BG) ===", flush=True)
    print(f"  min={img_raw.min().item():.4f} max={img_raw.max().item():.4f} mean={img_raw.mean().item():.4f}", flush=True)

    # 2) BilateralGrid only
    rgb_hwc = img_raw.permute(1, 2, 0).contiguous()
    out_bg = bg_grid(rgb_hwc, 0)
    img_bg = out_bg.permute(2, 0, 1).contiguous()
    print(f"\n=== BilateralGrid only ===", flush=True)
    print(f"  min={img_bg.min().item():.4f} max={img_bg.max().item():.4f} mean={img_bg.mean().item():.4f}", flush=True)
    diff_bg = (img_bg - img_raw).abs().mean().item()
    print(f"  mean abs diff from RAW: {diff_bg:.4f}", flush=True)

    # 3) PPISP only (apply after raw)
    rgb_hwc = img_raw.permute(1, 2, 0).contiguous()
    out_ppisp = ppisp(rgb_hwc, frame_idx=0, camera_idx=0)
    img_ppisp = out_ppisp.permute(2, 0, 1).contiguous()
    print(f"\n=== PPISP only ===", flush=True)
    print(f"  min={img_ppisp.min().item():.4f} max={img_ppisp.max().item():.4f} mean={img_ppisp.mean().item():.4f}", flush=True)
    diff_ppisp = (img_ppisp - img_raw).abs().mean().item()
    print(f"  mean abs diff from RAW: {diff_ppisp:.4f}", flush=True)

    # 4) BilateralGrid + PPISP
    rgb_hwc = img_bg.permute(1, 2, 0).contiguous()
    out_both = ppisp(rgb_hwc, frame_idx=0, camera_idx=0)
    img_both = out_both.permute(2, 0, 1).contiguous()
    print(f"\n=== BilateralGrid + PPISP ===", flush=True)
    print(f"  min={img_both.min().item():.4f} max={img_both.max().item():.4f} mean={img_both.mean().item():.4f}", flush=True)

    # 5) GT
    gt = vc.original_image.cuda()
    print(f"\n=== GT ===", flush=True)
    print(f"  min={gt.min().item():.4f} max={gt.max().item():.4f} mean={gt.mean().item():.4f}", flush=True)

    # Save all four for visual comparison
    def save(img, name):
        n = img.detach().cpu().numpy().transpose(1, 2, 0)
        n = np.clip(n * 255, 0, 255).astype(np.uint8)
        Image.fromarray(n).save(f"/home/jobs/20260911093928-f738e7bc/{name}.png")
    save(img_raw, "compare_raw")
    save(img_bg, "compare_bg")
    save(img_ppisp, "compare_ppisp")
    save(img_both, "compare_both")
    save(gt, "compare_gt")
    print("\n[saved] compare_{raw,bg,ppisp,both,gt}.png to job dir", flush=True)

    # Check PPISP actual params (after init, identity)
    print(f"\n[ppisp params after init]", flush=True)
    print(f"  exposure.raw: mean={ppisp.exposure.raw.mean().item():.4f}", flush=True)
    print(f"  color.s: mean={ppisp.color.s.mean().item():.4f}", flush=True)
    if hasattr(ppisp, "crf"):
        for k, v in ppisp.crf.named_parameters():
            print(f"  crf.{k}: shape={tuple(v.shape)} mean={v.mean().item():.4f}", flush=True)
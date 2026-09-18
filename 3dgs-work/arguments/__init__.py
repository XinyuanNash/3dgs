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

from argparse import ArgumentParser, BooleanOptionalAction, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action=BooleanOptionalAction)
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    # P1-5+ (2026-08-04): switch from store_true to
                    # BooleanOptionalAction so default-True bool fields
                    # (e.g. igs_plus_use_edge) are user-opt-out-able via
                    # ``--no-igs-plus-use-edge``. Side effect: every bool
                    # field in OptimizationParams/ModelParams/PipelineParams
                    # now exposes both ``--X`` and ``--no-X`` (positive form
                    # accepted as identity, negative form flips default).
                    group.add_argument("--" + key, default=value, action=BooleanOptionalAction)
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._depths = ""
        self._resolution = -1
        self._white_background = False
        self.train_test_exp = False
        self.data_device = "cuda"
        self.eval = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.antialiasing = True
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.025
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.exposure_lr_init = 0.01
        self.exposure_lr_final = 0.001
        self.exposure_lr_delay_steps = 0
        self.exposure_lr_delay_mult = 0.0
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 25_000  # P1-5.5: align with LFS (was 15_000)
        self.densify_grad_threshold = 0.0002
        # P1-MultiStrategy (OPTIMIZATIONS.md §13.2): strategy selector.
        # "default" preserves the original 3DGS clone/split/prune behavior
        # bit-identically. mcmc / igs_plus land in P1-3 / P1-4. The flag is
        # added in P1-1 so the strategy infrastructure has a stable CLI
        # surface from the start; non-default values raise NotImplementedError
        # until their corresponding commits land.
        self.densification_strategy = "default"
        # P1-MultiStrategy (OPTIMIZATIONS.md §13.6 + §13.7): MCMC knobs.
        # max_cap > 0 + densification_strategy="default" triggers an auto-
        # switch to "mcmc" in train.py (mirrors LFS convention where
        # --max-cap >= 0 implies MCMC). Set --densification_strategy mcmc
        # explicitly to force MCMC even without a cap.
        #
        # mcmc_noise_lr NOTE (calibrated per OPTIMIZATIONS.md §13.6
        # "scheduler-aware noise injection", 2026-07-27 supervisor
        # decision): the LFS reference value is 5e5 but LFS uses its own
        # per-step ExponentialLR scheduler (lr *= gamma^N). 3dgs-work
        # uses position_lr_max_steps-based decay instead, so lr stays
        # near lr_init for the early iterations. The 3dgs-calibrated
        # default is 1e5 (vs LFS's 5e5); 800-iter campus + 18K × 1
        # campus validation gates set the threshold per §13.9.
        # [P1-LFSTrainAlign-revert]: mcmc_* values restored to pre-LFSTrainAlign
        # (3dgs-original) defaults. LFSTrainAlign's 1.6e-5 means_lr + 5e5 noise_lr
        # combo caused -8.04 dB on campus 30K (see
        # output_campus_30k_three_way_REPORT.md §1.2).
        self.max_cap                 = -1     # absolute max Gaussian count (-1 = no cap)
        # [P1-MCMCParity-resume 2026-08-10] default 0.05 → 0.005 per
        # P1-MCMC-LFS-parity-spec.md §0.5 (LFS min_opacity = 0.005f).
        # 0.05 is the 3dgs-original default; the previous spec section was
        # never applied. See STATUS_P1-MCMCParity-resume.md §3 for
        # gap-closing rationale + 18K validation expectation.
        self.mcmc_opacity_threshold  = 0.005  # sigmoid(opacity) <= this -> dead (LFS default)
        self.mcmc_min_cap            = 0      # absolute min alive count (LFS default 0)
        self.mcmc_relocate_every     = 100    # refine cadence (iterations); LFS default 100
        self.mcmc_noise_lr           = 1e5    # 3dgs-calibrated noise scale (see comment above); LFS ref = 5e5 against its own scheduler (DO NOT copy)
        # === Path A (P1-MCMC-PathA, 2026-08-13 supervisor approval): opt-in flag
        # to disable noise injection entirely. Per Ablation E diagnostic, noise
        # LR scale mismatch (3dgs LR 1.6e-4 vs LFS LR 1.6e-5) causes noise to
        # destroy opacity Gaussians during the plateau phase. Default OFF
        # (False) preserves existing behavior; users opt in via
        # --mcmc_disable_noise. Densification (clone/relocate) is unaffected —
        # only _inject_noise is skipped.
        self.mcmc_disable_noise      = False
        # === Path C P0 #1 (P1-MCMC-PathC, 2026-08-14 supervisor approval): opt-in
        # flag to disable periodic opacity reset during MCMC densification.
        # LFS MCMC has NO opacity reset (reset exists only in
        # ImprovedGSPlus::reset_opacity, not the MCMC path). 3dgs calls
        # m.reset_opacity() every opacity_reset_interval=3000 iters
        # (mcmc.py:608-610), which clamps all opacities to <=0.01 and
        # destroys high-opacity Gaussians. Estimated Δ +1.0 to +3.0 dB on
        # data5 at 30K (audit: doc/status/STATUS_P1-MCMC-PathC-Audit.md
        # Top-5 #1). Default OFF preserves existing behavior; users opt in
        # via --mcmc_disable_opacity_reset. Bit-identical when OFF.
        self.mcmc_disable_opacity_reset = False
        # === Path C P0 #2 (P1-MCMC-PathC, 2026-08-14 supervisor approval): opt-in
        # flag to use LFS-aligned quaternion normalization in MCMC noise injection.
        # 3dgs mcmc.py:680-682 uses `q_norm = clamp(sqrt(dot), min=1e-6)` which
        # produces HUGE normalized q when `dot < 1e-12` (Gaussian's rotation is
        # nearly degenerate). LFS mcmc_kernels.cu:139 uses
        # `inv_norm = fminf(rsqrt(dot), 1e12f)` which CAPS the reciprocal at 1e12
        # (so degenerate q → near-zero normalized q, NOT huge). The 3dgs form
        # amplifies noise on near-degenerate q; LFS bounds it. Audit:
        # doc/status/STATUS_P1-MCMC-PathC-Audit.md Top-5 #2. Default OFF
        # preserves existing behavior (bit-identical when OFF).
        self.mcmc_lfs_quat_norm     = False
        # P1-MCMCParity-C arguments (D-segment opt-in + n_max + tail) — KEPT
        # (these are POST-LFSTrainAlign additions, NOT part of the revert):
        self.mcmc_relocate_n_max    = 51        # LFS _n_max (mcmc.hpp:87)
        self.sh_degree_interval     = 1000      # LFS default (parameters.hpp:90)
        self.freeze_every           = 0         # opt-in, off by default
        # D.8 noise cadence: 0 = legacy semantics (stop noise at
        # densify_until_iter). >0 extends the noise window by N iters
        # past densify_until_iter. Spec §5.3 default 5000 was rejected by
        # 18K gate (campus -5.96 dB regression); default 0 keeps bit-
        # equivalence to the pre-C LFSTrainAlign+A+B baseline; users
        # opt in explicitly via --noise_tail_iters N.
        self.noise_tail_iters       = 0         # post-densify_until_iter tail (0 = off)
        # P1-7+ (OPTIMIZATIONS.md §13.6): enable the per-Gaussian pixel-error
        # buffer path in the rasterizer. When True (and densification_strategy
        # is "mcmc"), MCMC's clone/relocate score uses the per-Gaussian
        # pixel-error instead of the viewspace grad-norm proxy. Default off
        # for backward compatibility; the 18K-campus gate sets the
        # threshold per §13.9.
        self.p1_7_plus               = False  # gate for new error buffer
        # === IGS+ Phase-1 knobs (P1-4-IGS+phase1; opt-in; does not affect
        # === other strategies unless --densification_strategy igs_plus).
        self.igs_plus_max_cap                = 1_000_000  # LFS schedule endpoint
        self.igs_plus_reset_every            = 3_000      # opacity reset cadence (LFS reset_every)
        self.igs_plus_opacity_prune_every    = 100        # mirrors densification_interval
        self.igs_plus_prune_opacity_threshold = 0.005     # LFS min_opacity
        self.igs_plus_score_floor            = 1e-12      # LFS densify_with_score clamp
        self.igs_plus_sh_degree_interval     = 1_000      # LFS parameters.hpp:90
        # P1-5 phase-2 gate (default ON 2026-08-04 per supervisor decision:
        # "走 C 直接默认 ON"; see [[p1-5-canny-data5-free]] for data5
        # verification that Canny is wall-clock-free on dense-coverage
        # scenes). When True, on_iteration_end runs kornia Canny + per-
        # Gaussian pixel sampling to augment the densification score
        # with edge-awareness. Opt-out via ``--no-igs-plus-use-edge``
        # (BooleanOptionalAction auto-generates the negative form).
        # CAVEAT (per [[p1-5-committed]]): on sparse-coverage scenes
        # (<50 imgs, full resolution) this gate was measured ~4× slower
        # at 18K campus; users on such scenes should opt out.
        self.igs_plus_use_edge               = True
        self.igs_plus_edge_n_cameras         = 10         # LFS random_cam_indices N min
        self.igs_plus_edge_min_cam_ratio     = 0.08       # LFS 8% dataset floor
        self.igs_plus_edge_score_weight      = 0.25       # LFS EDGE_SCORE_WEIGHT
        self.igs_plus_canny_low_threshold    = 0.1        # kornia.canny low
        self.igs_plus_canny_high_threshold   = 0.2        # kornia.canny high
        # P1-5.7 scale regularizer (default OFF; opt-in via --igs_plus_scale_reg_weight).
        # LFS fused_scale_regularization_kernel equivalent:
        #   loss += weight * mean(exp(scaling_raw))
        # LFS documented default: 0.01 (parameters.hpp:XXX). We default 0.0 for
        # bit-identity with P1-5.6; user opts in by passing > 0.
        self.igs_plus_scale_reg_weight       = 0.0
        # === P2-ADMM (Gaussian opacity sparsification) ===
        # Consensus ADMM-lite over opacity only. x-step is NOT modified
        # (no extra gradient) — only z (proximal hard-threshold) and u
        # (dual) are updated every ``admm_step_every`` iters starting at
        # ``admm_start_iter``. At ``admm_end_iter`` (or end of training)
        # ``prune_points(z == 0)`` physically removes the dead Gaussians.
        # See ``utils/sparsity.py:ADMMController`` for the math.
        # Default: ALL OFF (rho=0 → disabled; bit-identical to pre-ADMM path).
        # User opts in by passing --admm_rho > 0 + --admm_lambda > 0.
        # LFS defaults (parameters.hpp + sparsity.cuh): rho=0.1, lambda=1e-4,
        # but we leave 0 by default so the loss + prune path stays off until
        # the supervisor enables it.
        self.admm_rho                       = 0.0
        self.admm_lambda                     = 0.0
        self.admm_start_iter                 = 30_000
        self.admm_step_every                 = 100
        self.admm_end_iter                   = 0     # 0 → opt.iterations
        # === P0-Bilateral-v2 (per-image appearance correction) ===
        # Opt-in via ``--use_bilateral_grid``. Default lr=5e-4 (NOT LFS
        # 2e-3) per [[p0-bilateral-failed-longrun]] §2 Fix #5: with the
        # 800-iter schedule, lr=2e-3 overshoots and regresses PSNR ~3 dB;
        # 5e-4 gives -0.04 dB vs no-bilateral baseline (within noise).
        # Default tv_weight=10 (matches LFS).
        #
        # v2 design: ``bilateral_grid_start_iter`` freezes the grid at
        # identity until that iter. v1 engaged from iteration 0 and the
        # grid drifted during densification (500-25K) on clean LLFF
        # datasets → -6 dB feedback loop. Default 26000 = densify_until
        # (25K) + 1000 buffer, so the failure window is eliminated. On
        # in-the-wild datasets with real per-image exposure drift
        # (data5), the grid engages AFTER densification completes and
        # learns to correct drift without destabilizing the schedule.
        self.use_bilateral_grid         = False
        self.bilateral_grid_X           = 16        # spatial W
        self.bilateral_grid_Y           = 16        # spatial H
        self.bilateral_grid_W           = 8         # luma bins L
        self.bilateral_grid_lr          = 5e-4
        self.tv_loss_weight             = 10.0
        self.bilateral_grid_start_iter  = 26000     # P0-Bilateral-v2: freeze gate
        # === P2-PPISP Phase 1 (4-stage differentiable ISP) ===
        # Port of LFS ppisp (CVPR 2025, NVIDIA): 4 stages — Exposure
        # (per-frame) → Vignetting (per-camera, 5-coeff poly) → Color
        # correction (per-frame, 8-dim latent → 3x3 homography) → CRF
        # (per-camera, 4-param piecewise power). All stages identity-init
        # so opt-in (`--use_ppisp`) is bit-identical to no PPISP at iter 0.
        # See ``utils/ppisp.py`` for math (mirrors
        # LichtFeld-Studio/src/training/kernels/ppisp_math.cuh).
        #
        # Defaults match LFS ppisp.hpp Config: lr=2e-3, Adam, warmup 500,
        # cosine decay to 0.01x. Per P0-Bilateral-v2 lesson, ``start_iter``
        # freezes the params at identity until that iter so the schedule
        # completes before PPISP starts learning. Default 26000 matches
        # bilateral_grid_start_iter (densify_until 25K + 1000 buffer).
        # Reg weights default to LFS ppisp.hpp values; disabled by setting
        # to 0.
        self.use_ppisp                       = False
        self.ppisp_lr                        = 2e-3
        self.ppisp_start_iter                = 26000
        self.ppisp_reg_exposure_mean         = 1.0    # smooth_l1(mean(exp), β=0.1)
        self.ppisp_reg_vig_center            = 0.02   # mean(cx² + cy²)
        self.ppisp_reg_vig_channel           = 0.1    # var over RGB channels
        self.ppisp_reg_vig_non_pos           = 0.01   # relu(α₀..α₂)
        self.ppisp_reg_color_mean            = 1.0    # smooth_l1(mean(color@pinv))
        self.ppisp_reg_crf_channel           = 0.1    # var over RGB channels of CRF
        self.depth_l1_weight_init = 1.0
        self.depth_l1_weight_final = 0.01
        self.random_background = False
        self.optimizer_type = "default"
        # === P3-Morton (Morton order for GPU cache locality) ===
        # Opt-in via ``--use_morton_order``. When True, train.py
        # periodically permutes Gaussians by Morton code of _xyz during
        # the densify window (every ``--morton_reorder_interval`` iters).
        # Pure data layout — does not affect loss / gradient; expected
        # PSNR delta is within ±0.01 dB (deterministic; same input →
        # same permutation). Goal: per-iter wall-clock -1% to -5% on
        # large datasets (~1M+ Gaussians) by improving rasterizer L1/L2
        # cache locality. Small datasets (N < ``--morton_min_gaussians``)
        # skip the reorder because overhead dominates the benefit.
        # See doc/specs/P3-Morton-spec.md for the full design.
        self.use_morton_order           = False   # master switch; default OFF
        self.morton_reorder_interval    = 100     # every N iters in densify window
        self.morton_min_gaussians       = 50000   # skip if N < this
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)

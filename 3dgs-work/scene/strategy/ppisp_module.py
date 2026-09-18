"""PPISPModule: wraps P2-PPISP 4-stage differentiable ISP (pure PyTorch).

Pipeline: Exposure → Vignetting → ColorCorrection → CRF.
Identity-init means: when frozen (iteration < start_iter), output == input
bit-identical. Engaged default OFF (``--use_ppisp``); default start_iter
26000 mirrors the P0-Bilateral-v2 freeze-during-densification pattern.

Unlike BilateralGrid (CUDA kernel), PPISP is pure PyTorch — no rasterizer
submodule import needed. See utils/ppisp.py for math.
"""
import torch

from scene.strategy.optional_module import OptInTrainModule, register


@register
class PPISPModule(OptInTrainModule):
    """P2-PPISP: 4-stage differentiable ISP (Exposure / Vignetting / Color / CRF).

    is_enabled   : getattr(opt, "use_ppisp", False)
    engaged      : iteration >= opt.ppisp_start_iter (default 26000)
    requires_scene: True (needs n_cams for per-frame + per-camera stages)

    zero_grad: mandatory each iter (NOT gated on engaged) to prevent
    PyTorch ``.grad`` accumulation across the 26K-iter frozen window.
    Without this, the first engaged Adam step receives the sum of all
    gradients → blows params into NaN. Mirrors LFS trainer.cpp pattern.
    """

    def __init__(self, opt, scene=None):
        self.opt = opt
        self.ppisp = None
        if not getattr(opt, "use_ppisp", False):
            return
        from utils.ppisp import PPISP as PPISPModel
        n_cams = len(scene.getTrainCameras()) + len(scene.getTestCameras())
        self.ppisp = PPISPModel(
            num_frames=n_cams,
            num_cameras=n_cams,  # 1 camera per frame in our COLMAP datasets
            total_iterations=opt.iterations,
            start_iter=opt.ppisp_start_iter,
            lr=opt.ppisp_lr,
            reg_weight_exposure_mean=opt.ppisp_reg_exposure_mean,
            reg_weight_vig_center=opt.ppisp_reg_vig_center,
            reg_weight_vig_channel=opt.ppisp_reg_vig_channel,
            reg_weight_vig_non_pos=opt.ppisp_reg_vig_non_pos,
            reg_weight_color_mean=opt.ppisp_reg_color_mean,
            reg_weight_crf_channel=opt.ppisp_reg_crf_channel,
        ).cuda()
        print(f"[P2-PPISP] engaged: num_frames={n_cams} num_cameras={n_cams} "
              f"lr={opt.ppisp_lr} start_iter={opt.ppisp_start_iter} (frozen until then) "
              f"reg=[exp={opt.ppisp_reg_exposure_mean} vig_c={opt.ppisp_reg_vig_center} "
              f"vig_ch={opt.ppisp_reg_vig_channel} vig_np={opt.ppisp_reg_vig_non_pos} "
              f"col={opt.ppisp_reg_color_mean} crf_ch={opt.ppisp_reg_crf_channel}]")

    @property
    def name(self) -> str:
        return "ppisp"

    @property
    def is_enabled(self) -> bool:
        return self.ppisp is not None

    @property
    def requires_scene(self) -> bool:
        return True  # n_cams for per-frame + per-camera stages

    def engaged(self, iteration: int) -> bool:
        if self.ppisp is None:
            return False
        return self.ppisp.engaged(iteration)

    def apply(self, rgb_chw: torch.Tensor, vind=0, iteration=None, **_) -> torch.Tensor:
        """4-stage ISP applied AFTER any bilateral correction, BEFORE L1/SSIM.

        Input: CHW float. Output: CHW float, ISP-corrected.
        When frozen (iteration < start_iter): return input bit-identical
        WITHOUT running the forward pass. Identity init is mathematically
        identity, but the 4-stage PyTorch pipeline (clamp/pow/exp edges)
        can still drop gradients at boundary pixels — empirically observed
        to collapse ABAB PSNR to ~5.6 dB by iter 7K when combined with
        --igs_plus_scale_reg_weight 0.01 (T8 / T10 bisect 2026-08-27).
        See STATUS_PPISP-applied-when-frozen.md.

        When engaged: full 4-stage forward runs, gradient flows normally.
        """
        if self.ppisp is None:
            return rgb_chw
        # Frozen window: skip the forward pass entirely. The Adam step is
        # also skipped by `step()` (gated by engaged), so the frozen window
        # has zero impact on params — the only thing that runs is zero_grad
        # (mandatory each iter to prevent .grad accumulation).
        # We mirror this in apply() so the image-transform chain doesn't
        # touch the gradient graph during the freeze window.
        if iteration is not None and not self.ppisp.engaged(iteration):
            return rgb_chw
        # CHW → HWC → ISP → CHW (matches train.py:277 pattern)
        rgb_hwc = rgb_chw.permute(1, 2, 0).contiguous()
        out = self.ppisp(rgb_hwc, frame_idx=vind, camera_idx=0)
        return out.permute(2, 0, 1).contiguous()

    def eval_apply(self, rgb_chw: torch.Tensor, vind=0, **_) -> torch.Tensor:
        """Re-apply in training_report. Caller wraps in torch.no_grad().

        No iteration kwarg threaded (eval only runs at saved iter points
        where PPISP has already engaged). When PPISP is engaged, the
        forward path runs normally — no gradient concern (caller is in
        ``torch.no_grad()``).
        """
        if self.ppisp is None:
            return rgb_chw
        return self.apply(rgb_chw, vind=vind)

    def reg_loss(self) -> torch.Tensor:
        """Sum of 6 weighted regularizers (exposure_mean, vig_center,
        vig_channel, vig_non_pos, color_mean, crf_channel).

        When opted out: returns 0 — no extra cost.
        """
        if self.ppisp is None:
            return torch.tensor(0.0, device="cuda")
        return self.ppisp.reg_loss()

    def step(self, iteration: int, **_) -> None:
        """Per-iter Adam step. Gated by `engaged(iteration)`."""
        if self.ppisp is not None and self.ppisp.engaged(iteration):
            self.ppisp.optimizer_step()

    def zero_grad(self) -> None:
        """Mandatory each iter (NOT gated on engaged).

        PyTorch ``.grad`` accumulates across backward() calls. The 26K-iter
        frozen window would let stale grads pile up and explode params on
        the first engaged step. Mirrors LFS trainer.cpp's universal
        zero_grad per iter.
        """
        if self.ppisp is not None:
            self.ppisp.zero_grad()

    def scheduler_step(self) -> None:
        """LR warmup/cosine schedule — always runs (universal loop)."""
        if self.ppisp is not None:
            self.ppisp.scheduler_step()
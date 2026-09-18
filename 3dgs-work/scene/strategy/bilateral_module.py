"""BilateralGridModule: wraps P0-Bilateral-v2 per-image appearance correction.

The grid is a per-camera affine correction trained via Adam that gates the
photometric loss through a learnable BilateralGrid (rasterizer submodule).
Frozen at identity until ``bilateral_grid_start_iter`` (default 26000)
to avoid the P0-Bilateral-v1 failure mode (drift during densification on
clean LLFF data — see [[p0-bilateral-failed-longrun]]).

Default OFF (``--use_bilateral_grid`` opt-in). When disabled, every method
is a no-op / identity so the universal loops skip the work entirely.
"""
import torch

from scene.strategy.optional_module import OptInTrainModule, register


@register
class BilateralGridModule(OptInTrainModule):
    """P0-Bilateral-v2: per-image appearance correction via BilateralGrid.

    is_enabled   : getattr(opt, "use_bilateral_grid", False)
    engaged      : iteration >= opt.bilateral_grid_start_iter
                   (default 26000 = densify_until 25K + 1000 buffer)
    requires_scene: True (needs ``scene.getTrainCameras() + getTestCameras()``
                   to size the grid)
    """

    def __init__(self, opt, scene=None):
        # Stash opt so step() can read tv_loss_weight without re-importing.
        self.opt = opt
        self.grid = None
        if not getattr(opt, "use_bilateral_grid", False):
            return
        # Lazy import: keeps the BilateralGrid CUDA-kernel submodule out of
        # the import graph when --use_bilateral_grid is OFF (default).
        # Mirrors pre-refactor train.py:44 — package name is
        # ``diff_gaussian_rasterization`` (set in submodules/.../setup.py:18),
        # NOT ``submodules.diff_gaussian_rasterization``.
        from diff_gaussian_rasterization import BilateralGrid
        n_cams = len(scene.getTrainCameras()) + len(scene.getTestCameras())
        self.grid = BilateralGrid(
            num_images=n_cams,
            grid_W=opt.bilateral_grid_X,
            grid_H=opt.bilateral_grid_Y,
            grid_L=opt.bilateral_grid_W,
            lr=opt.bilateral_grid_lr,
            total_iterations=opt.iterations,
            tv_weight=opt.tv_loss_weight,
            start_iter=opt.bilateral_grid_start_iter,
        )
        print(f"[P0-Bilateral-v2] engaged: num_images={n_cams} "
              f"grid=({opt.bilateral_grid_X},{opt.bilateral_grid_Y},{opt.bilateral_grid_W}) "
              f"lr={opt.bilateral_grid_lr} tv_weight={opt.tv_loss_weight} "
              f"start_iter={opt.bilateral_grid_start_iter} (frozen until then)")

    @property
    def name(self) -> str:
        return "bilateral_grid"

    @property
    def is_enabled(self) -> bool:
        return self.grid is not None

    @property
    def requires_scene(self) -> bool:
        return True  # n_cams comes from Scene

    def engaged(self, iteration: int) -> bool:
        if self.grid is None:
            return False
        return self.grid.engaged(iteration)

    def apply(self, rgb_chw: torch.Tensor, vind=0, iteration=None, **_) -> torch.Tensor:
        """Bilateral grid correction applied BETWEEN render() and L1/SSIM.

        Input: CHW float (rasterizer native layout).
        Output: CHW float, bilaterally-corrected.
        Contract: when frozen (engaged=False), return input bit-identical
        WITHOUT running the forward pass.

        Root cause (2026-09-15 investigation): the installed
        ``diff_gaussian_rasterization`` (site-packages, P1-7+ revision)
        wraps ``_C.bilateral_grid_forward`` with a plain ``* self.color_scale``
        — there is NO ``torch.autograd.Function`` between the C++ kernel
        and the output tensor. PyTorch then synthesizes a ``MulBackward0``
        for the color-scale mult, but the upstream ``out_hwc`` from
        ``_C.bilateral_grid_forward`` has no grad_fn → grad cannot flow
        back to ``rgb`` → image-loss gradient dies at this layer.

        Empirically the failure is invisible in PHOTOMETRIC terms (output
        is identity when grid is initialized as ones), but
        ``loss.backward()`` only populates ``_scaling.grad`` (via
        scale_reg), leaving 5/6 Gaussian params with ``grad is None`` —
        the optimizer's lazy state init then sees ``param.shape=(22351, 3)``
        with no Adam state at densification iter 500 → RuntimeError.

        The fix mirrors PPISPModule: skip the forward entirely during
        the frozen window. The grid Adam step is also gated by
        ``engaged(iteration)`` so the frozen window has zero side effects.
        """
        if self.grid is None:
            return rgb_chw
        # Frozen window: skip the forward pass entirely. Adam step is
        # also skipped by ``step()`` (gated by engaged), so the frozen
        # window has zero impact on params — the only thing that runs
        # is zero_grad (mandatory each iter to prevent .grad accumulation).
        if iteration is not None and not self.grid.engaged(iteration):
            return rgb_chw
        # CHW → HWC → grid → CHW (matches train.py:266 pattern)
        rgb_hwc = rgb_chw.permute(1, 2, 0).contiguous()
        out = self.grid(rgb_hwc, vind)
        return out.permute(2, 0, 1).contiguous()

    def eval_apply(self, rgb_chw: torch.Tensor, vind=0, **_) -> torch.Tensor:
        """Re-apply in training_report. Caller wraps in torch.no_grad()."""
        if self.grid is None:
            return rgb_chw
        return self.apply(rgb_chw, vind=vind)

    def step(self, iteration: int, **_) -> None:
        """Per-iter Adam step. Gated by `engaged(iteration)` (universal loop).

        tv_backward BEFORE optimizer_step — matches LFS trainer.cpp order
        and train.py:390-392.
        """
        if self.grid is None:
            return
        if self.opt.tv_loss_weight > 0:
            self.grid.tv_backward(self.opt.tv_loss_weight)
        self.grid.optimizer_step()

    def scheduler_step(self) -> None:
        """LR warmup/exp-decay schedule — always runs (universal loop)."""
        if self.grid is not None:
            self.grid.scheduler_step()
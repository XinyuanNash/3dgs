"""Default densification strategy — mirror of the original 3DGS densification
behavior (gradient-thresholded clone + LAS split + opacity prune + periodic
opacity reset).

This is the strategy that 3dgs-work has had since P0-LAS. The goal of P1-1 is
to **refactor without behavior change**: this class must be 1:1 equivalent to
the inline block that used to live in ``train.py:163-174``. Verified by 800-iter
× 3 campus determinism + PSNR diff ≤ 0.01 dB vs pre-refactor baseline.

Reference (the block being replaced):
    if iteration < opt.densify_until_iter:
        gaussians.max_radii2D[visibility_filter] = torch.max(
            gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
        gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
        if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
            size_threshold = 20 if iteration > opt.opacity_reset_interval else None
            gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005,
                                        scene.cameras_extent, size_threshold, radii)
        if iteration % opt.opacity_reset_interval == 0 or (
                dataset.white_background and iteration == opt.densify_from_iter):
            gaussians.reset_opacity()
"""
import torch

from .base import Strategy
from .factory import register


@register("default")
class DefaultStrategy(Strategy):
    """Original 3DGS densification, refactored behind the Strategy interface."""

    #: Hardcoded ``min_opacity`` prune threshold (was a magic ``0.005`` literal
    #: in train.py:171). Promoted to an instance attribute so future strategies
    #: (e.g. IGS+) can override the value without touching this class.
    min_opacity: float = 0.005

    #: Hardcoded ``size_threshold`` pixels — Gaussians larger than this are
    #: pruned in screen-space. ``None`` disables screen-size pruning (used
    #: before ``opacity_reset_interval`` iterations have elapsed).
    size_threshold_pixels: int = 20

    @property
    def name(self) -> str:
        return "default"

    def post_backward(
        self,
        iteration: int,
        viewspace_point_tensor,
        visibility_filter,
        radii,
        scene_extent: float,
        error_buffer=None,  # P1-7+: unused by DefaultStrategy
    ) -> None:
        opt = self.opt
        m = self.model

        if iteration >= opt.densify_until_iter:
            return

        # 1. Update max_radii2D from current frame's radii
        m.max_radii2D[visibility_filter] = torch.max(
            m.max_radii2D[visibility_filter], radii[visibility_filter]
        )

        # 2. Accumulate xyz grad for the clone/split decision
        m.add_densification_stats(viewspace_point_tensor, visibility_filter)

        # 3. Periodic densify_and_prune (clone + LAS split + opacity prune)
        if (
            iteration > opt.densify_from_iter
            and iteration % opt.densification_interval == 0
        ):
            size_threshold = (
                self.size_threshold_pixels
                if iteration > opt.opacity_reset_interval
                else None
            )
            m.densify_and_prune(
                opt.densify_grad_threshold,
                self.min_opacity,
                scene_extent,
                size_threshold,
                radii,
            )

        # 4. Periodic opacity reset
        white_bg = (
            self.dataset is not None
            and getattr(self.dataset, "white_background", False)
        )
        if (
            iteration % opt.opacity_reset_interval == 0
            or (white_bg and iteration == opt.densify_from_iter)
        ):
            m.reset_opacity()

    def step(self, iteration: int) -> None:
        # No-op: the default strategy has no per-step work after the optimizer.
        pass

    def is_refining(self, iteration: int) -> bool:
        return iteration < self.opt.densify_until_iter

    def on_iteration_end(self, iteration: int, scene_extent: float, radii) -> None:
        # No-op: the default strategy has no end-of-iteration hook.
        pass

    def state_dict(self) -> dict:
        # The default strategy has no per-iteration state worth checkpointing
        # (densification is purely a function of the GaussianModel's buffers
        # and the iteration counter, both of which are restored elsewhere).
        return {}

    def load_state_dict(self, sd: dict) -> None:
        # No-op: see state_dict().
        pass
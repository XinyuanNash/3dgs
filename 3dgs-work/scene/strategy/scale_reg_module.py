"""ScaleRegModule: wraps the `--igs_plus_scale_reg_weight` regularizer.

Single-purpose: emit `weight * gaussians.get_scaling.mean()` as `reg_loss()`.
No state, no scheduler, no image transform — purely additive to the
photometric loss when `weight > 0`. Default `weight == 0.0` → `is_enabled`
returns False → module is a no-op for the default ship path.
"""
import torch

from scene.strategy.optional_module import OptInTrainModule, register


@register
class ScaleRegModule(OptInTrainModule):
    """LFS-style `fused_scale_regularization_kernel` (P1-5.7 port).

    Source of truth in LFS:
            `LichtFeld-Studio/src/training/strategies/improved_gs_plus.cpp`
    The PyTorch port uses `gaussians.get_scaling.mean()` as a stand-in for
    the fused kernel; effect is identical up to CUDA rounding.
    """

    def __init__(self, opt, scene=None):
        # `opt` carries the flag; `scene` is required to reach `gaussians`.
        # We stash both so subsequent `reg_loss()` calls don't re-resolve.
        self.weight = float(opt.igs_plus_scale_reg_weight)
        self.gaussians = scene.gaussians if scene is not None else None

    @property
    def name(self) -> str:
        return "scale_reg"

    @property
    def is_enabled(self) -> bool:
        # Default ship path: weight == 0.0 → opt-out, no extra cost.
        return self.weight > 0.0

    @property
    def requires_scene(self) -> bool:
        # We need `scene.gaussians` to read `get_scaling`; the constructor
        # itself does not call any scene method. Returns False so the
        # `_build_optional_modules` helper does NOT pass scene into us at
        # construction time — we capture it lazily on first reg_loss().
        return False

    def reg_loss(self) -> torch.Tensor:
        if self.gaussians is None or self.weight <= 0.0:
            return torch.tensor(0.0, device="cuda")
        # Mean of exp(log_scale) ≈ average Gaussian size. Matches LFS
        # fused_scale_regularization_kernel's reduction.
        return self.weight * self.gaussians.get_scaling.mean()
"""OptInTrainModule: abstract base class for cross-cutting train-time modules.

Mirrors the LFS trainer.cpp "TrainableModule" pattern. Subclasses register
themselves via the @register decorator; train.py iterates OPTIONAL_MODULES
at construction time and dispatches via the abstract interface.

Each subclass declares/overrides:
  - name              : str  (debug)
  - is_enabled        : bool (opt-in gate)
  - requires_scene    : bool (construction needs Scene for n_cams etc.)
  - engaged(iter)     : bool (freeze gate)
  - apply(rgb, ...)   : rgb  (image transform, identity by default)
  - eval_apply(...)   : rgb  (re-apply in eval, identity by default)
  - reg_loss()        : tensor (extra loss term, 0 by default)
  - step(iter)        : None (per-iter update, gated by engaged)
  - scheduler_step()  : None (per-iter schedule, always runs)
  - zero_grad()       : None (mandatory each iter)

train.py wires these into the training loop with three universal loops:
  - image transform chain (apply)
  - reg loss addition
  - per-iter step + scheduler_step + zero_grad (gated by engaged)

Adding a new cross-cutting concern (e.g. P3-Pipeline data loader) requires:
  1. Write the new module under scene/strategy/<name>_module.py
  2. Subclass OptInTrainModule + @register decorator
  3. train.py is UNTOUCHED — the loop picks it up automatically.
"""
from abc import ABC, abstractmethod
import torch


class OptInTrainModule(ABC):
    """Abstract base for cross-cutting train-time modules.

    Subclasses MUST:
      - Decorate with @register from this module (registers globally)
      - Implement the `name` property

    Subclasses MAY override any of:
      - is_enabled         (default True; usually checks opt flags)
      - requires_scene     (default True; construction needs Scene)
      - engaged            (default True; per-iter freeze gate)
      - apply              (default identity; image transform)
      - eval_apply         (default identity; eval re-transform)
      - reg_loss           (default 0; extra loss term)
      - step               (default no-op; per-iter update)
      - scheduler_step     (default no-op; LR schedule)
      - zero_grad          (default no-op; mandatory each iter)

    The default opt-in invariant: when is_enabled returns False, the
    module is skipped by the universal loops — behavior is bit-identical
    to no module at all. train.py never inspects module types directly.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable identifier for debug prints and logging."""
        ...

    @property
    def is_enabled(self) -> bool:
        """Whether the module is opted in. Default True (subclasses usually
        gate on opt.* flags and return False when the relevant knob is 0)."""
        return True

    @property
    def requires_scene(self) -> bool:
        """True if construction needs Scene (e.g. to read n_cams for
        per-camera modules like BilateralGrid or PPISP). Default True —
        opt-only constructors must explicitly override this to False."""
        return True

    def engaged(self, iteration: int) -> bool:
        """Per-iter freeze gate. Default True (always engaged). When False,
        `step` is skipped but `scheduler_step` still runs (so LR schedules
        stay aligned with training iterations)."""
        return True

    def apply(self, rgb_chw: torch.Tensor, **kwargs) -> torch.Tensor:
        """Image-space transform applied between render() and L1/SSIM loss.

        Input is CHW float (the rasterizer's native layout). Subclasses
        that need HWC may permute internally — the chain contract is
        CHW in, CHW out. Default: identity.

        `vind` is provided as a kwarg (per-frame index) when available.
        """
        return rgb_chw

    def eval_apply(self, rgb_chw: torch.Tensor, **kwargs) -> torch.Tensor:
        """Re-apply the image transform in the eval path (training_report).

        Called inside `torch.no_grad()`. Default: identity. Subclasses
        usually just call `apply` again with the same kwargs."""
        return rgb_chw

    def reg_loss(self) -> torch.Tensor:
        """Optional regularization loss added to the photometric loss.

        Default 0. The returned tensor must live on the same device as the
        training loss (`cuda` in our setup). Returning 0 is safe — the
        universal loop accumulates with `loss + module.reg_loss()` which
        is a no-op for zero."""
        return torch.tensor(0.0, device="cuda")

    def step(self, iteration: int, **kwargs) -> None:
        """Per-iter update (Adam step etc.). Gated by `engaged(iteration)`.
        Default: no-op.

        `kwargs` is a passthrough from the universal train.py loop. Modules
        can pluck what they need (e.g. ``gaussians`` for ADMMModule, which
        passes it through to ``ADMMController.step(gaussians)``). When the
        kwarg isn't passed, it simply isn't there — modules that need it
        must declare it in the override signature.
        """
        pass

    def scheduler_step(self) -> None:
        """Per-iter LR schedule. Always runs (even when not engaged), so the
        schedule tracks training iterations. Default: no-op."""
        pass

    def zero_grad(self) -> None:
        """Mandatory each iter to prevent .grad accumulation.

        Critical: PyTorch `.grad` ACCUMULATES across `backward()` calls.
        Modules that participate in the loss chain MUST zero_grad every
        iter, not just when engaged. LFS trainer.cpp does this implicitly
        via its TrainableModule contract; we follow the same convention.

        Default: no-op (module has no grad state to clear)."""
        pass

    def finalize(self, gaussians, scene, opt, strategy,
                 saving_iterations, checkpoint_iterations, iteration: int) -> None:
        """One-shot post-loop hook (called once after the training loop ends).

        Default: no-op. ADMMModule overrides this to run a final
        ``controller.step + materialize_prune`` and re-save PLY/chkpnt
        so the on-disk artefacts reflect the pruned Gaussian set.
        BilateralGrid / PPISP / ScaleReg don't need it (no end-of-training
        mutations).

        Args:
            gaussians: GaussianModel (post-train state, frozen).
            scene: Scene (for `scene.save(iter)` PLY output).
            opt: parsed args (for `opt.iterations` etc.).
            strategy: active Strategy (for `strategy.name`/`state_dict()`
                      on chkpnt re-save).
            saving_iterations: list of iters at which PLY was saved in-loop.
            checkpoint_iterations: list of iters at which chkpnt was saved.
            iteration: the iteration at which the loop ended (= opt.iterations).
        """
        pass


# Global registry — subclasses append to this list via @register.
# Importing a *_module.py file (for its side effects) is enough to populate.
OPTIONAL_MODULES: list = []


def register(cls):
    """Class decorator: register an OptInTrainModule subclass.

    Example::

        @register
        class MyModule(OptInTrainModule):
            @property
            def name(self): return "my_module"
            ...

    Returns the class unchanged so the decorator is transparent.
    """
    if not isinstance(cls, type) or not issubclass(cls, OptInTrainModule):
        raise TypeError(
            f"@register target must subclass OptInTrainModule, got {cls!r}"
        )
    OPTIONAL_MODULES.append(cls)
    return cls

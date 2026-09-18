"""ADMMModule: wraps P2-ADMM consensus ADMM-lite (Gaussian opacity sparsify).

Default OFF (both ``opt.admm_rho > 0`` and ``opt.admm_lambda > 0`` required).
When enabled, ADMMController takes a step every ``admm_step_every`` iters
between ``admm_start_iter`` and ``admm_end_iter``, then physically prunes
Gaussians where the proximal operator decided z == 0 at end-of-training.

The wrapper delegates ALL math/state to ADMMController (utils/sparsity.py);
we only adapt the controller's interface to OptInTrainModule's `step` /
`finalize` hooks so the trainer can drive it via the universal loops.

Source of truth: P2-ADMM commit `6f0b0f0` (train.py:182-194, 322-330, 440-462).
"""
import torch

from scene.strategy.optional_module import OptInTrainModule, register


@register
class ADMMModule(OptInTrainModule):
    """P2-ADMM: consensus ADMM-lite for Gaussian opacity sparsification.

    is_enabled   : opt.admm_rho > 0 and opt.admm_lambda > 0
    engaged      : opt.admm_start_iter <= iteration (no upper bound;
                   finalize() handles end-of-training prune)
    requires_scene: False (only needs `opt` and `gaussians` at step time)
    """

    def __init__(self, opt, scene=None):
        # Stash `opt` for end-of-training resolve_end_iter + re-save gating.
        self.opt = opt
        # Construct controller lazily: only when actually enabled. This
        # preserves the bit-identical behaviour of train.py:185-194
        # (`if opt.admm_rho > 0 and opt.admm_lambda > 0: ADMMController(opt)`).
        self.controller = None
        if opt.admm_rho > 0 and opt.admm_lambda > 0:
            from utils.sparsity import ADMMController
            self.controller = ADMMController(opt)
            self.controller_end_iter = self.controller.resolve_end_iter(opt.iterations)
            print(f"[P2-ADMM] engaged: rho={opt.admm_rho} lambda={opt.admm_lambda} "
                  f"start={opt.admm_start_iter} step_every={opt.admm_step_every} "
                  f"end={self.controller_end_iter}")

    @property
    def name(self) -> str:
        return "admm"

    @property
    def is_enabled(self) -> bool:
        return self.controller is not None

    @property
    def requires_scene(self) -> bool:
        return False  # only needs gaussians at step/finalize time

    def engaged(self, iteration: int) -> bool:
        # ADMMController.engaged() already gates on rho > 0, lambda > 0,
        # and iteration >= start_iter. We don't gate on end_iter here —
        # the trainer still wants `step` calls inside the [start, end]
        # window, and finalize() picks up the prune at end_iter.
        if self.controller is None:
            return False
        return self.controller.engaged(iteration)

    def step(self, iteration: int, **kwargs) -> None:
        # ADMM is gated on `iteration % opt.admm_step_every == 0` per
        # train.py:329 — bind it here so the universal loop doesn't have to
        # know about ADMM's stride.
        if iteration % self.opt.admm_step_every != 0:
            return
        gaussians = kwargs.get("gaussians")
        if gaussians is None:
            # Universal-loop invariant violation; safer to no-op than crash.
            return
        self.controller.step(gaussians)

    def finalize(self, gaussians, scene, opt, strategy,
                 saving_iterations, checkpoint_iterations, iteration: int) -> None:
        """End-of-training prune + re-save (mirrors train.py:440-462).

        The in-loop ``scene.save(iteration)`` / ``torch.save(...)`` at
        opt.iterations already captured the pre-prune state; we re-save
        here so the on-disk artefacts reflect the pruned Gaussian set.
        chkpnt is only re-saved if the user opted in via
        ``--checkpoint_iterations`` to avoid silent overwrites.
        """
        if self.controller is None:
            return
        if not self.controller.engaged(iteration):
            return
        # Final step + materialize prune.
        self.controller.step(gaussians)
        n_pruned = self.controller.materialize_prune(gaussians)
        print(f"[P2-ADMM] pruned {n_pruned} Gaussians (z==0); "
              f"remaining = {gaussians.get_xyz.shape[0]}")
        # Re-save PLY at the final iter to capture the pruned state.
        if iteration in saving_iterations:
            print(f"[P2-ADMM] re-saving PLY at iter {iteration} "
                  f"to reflect pruned set (was pre-prune)")
            scene.save(iteration)
        # Re-save chkpnt if the user requested one at the final iter.
        if iteration in checkpoint_iterations:
            print(f"[P2-ADMM] re-saving chkpnt at iter {iteration} "
                  f"to reflect pruned set")
            torch.save(
                (
                    gaussians.capture(),
                    strategy.name,
                    strategy.state_dict(),
                    iteration,
                ),
                scene.model_path + "/chkpnt" + str(iteration) + ".pth",
            )
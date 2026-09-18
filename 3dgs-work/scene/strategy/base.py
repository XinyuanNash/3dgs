"""Strategy abstract base class.

See OPTIMIZATIONS.md §13.4 for the full spec.

A Strategy owns the densification policy for a GaussianModel: it decides when
to clone / split / prune / relocate Gaussians, and when to reset opacity.
Different strategies (default, MCMC, IGS+, …) implement the same interface so
that ``train.py`` can swap them via ``--densification_strategy``.

The base class is abstract; concrete implementations live in sibling modules
(defaults.py, mcmc.py, igs_plus.py).
"""
from abc import ABC, abstractmethod
from typing import Any, Dict

import torch


class Strategy(ABC):
    """Abstract densification strategy.

    Args:
        model:   The :class:`scene.gaussian_model.GaussianModel` to operate on.
                 The strategy reads/writes densification buffers (max_radii2D,
                 xyz_gradient_accum, denom, …) and may call clone / split /
                 prune / reset_opacity / optimizer-state surgery helpers.
        opt:     The parsed :class:`arguments.OptimizationParams`. All
                 densification hyperparameters live here (``densify_from_iter``,
                 ``densify_until_iter``, ``densification_interval``,
                 ``densify_grad_threshold``, ``percent_dense``,
                 ``opacity_reset_interval``, …).
        dataset: Optional reference to the loaded dataset (a ``Scene`` /
                 ``ModelParams`` object). Needed only by strategies that
                 inspect ``dataset.white_background`` (currently just
                 ``DefaultStrategy``). May be ``None``.
    """

    def __init__(self, model, opt, dataset=None):
        self.model = model
        self.opt = opt
        self.dataset = dataset

    @abstractmethod
    def post_backward(
        self,
        iteration: int,
        viewspace_point_tensor,
        visibility_filter,
        radii,
        scene_extent: float,
        error_buffer=None,
    ) -> None:
        """Called once per training iteration, *after* ``loss.backward()``
        and *before* the optimizer step. This is the densification trigger
        point (mirrors LFS ``IStrategy::post_backward``).

        DefaultStrategy uses it to:
            1. Update ``max_radii2D`` from the current frame's radii.
            2. Accumulate ``xyz_gradient_accum`` / ``denom`` for the
               clone-or-split decision.
            3. Every ``densification_interval`` steps (after
               ``densify_from_iter``), call
               ``model.densify_and_prune(...)`` which clones small high-grad
               Gaussians, splits large high-grad Gaussians via LAS, and
               prunes low-opacity / over-large ones.
            4. Periodically reset opacity.

        ``error_buffer`` (P1-7+, OPTIMIZATIONS.md §13.6) is the rasterizer's
        persistent per-Gaussian pixel-error buffer when ``--p1_7_plus`` is
        set; otherwise None. Strategies that consume it (MCMC) can fall back
        to their legacy proxy when it's missing.
        """

    @abstractmethod
    def step(self, iteration: int) -> None:
        """Called once per training iteration, *after*
        ``gaussians.optimizer.step()``. DefaultStrategy has nothing to do
        here; MCMC injects per-step noise on ``_xyz``; IGS+ runs edge-score
        pre-passes asynchronously."""

    @abstractmethod
    def is_refining(self, iteration: int) -> bool:
        """True iff this iteration is inside the densification window."""

    @abstractmethod
    def on_iteration_end(self, iteration: int, scene_extent: float, radii) -> None:
        """Hook called at the *end* of an iteration (after optimizer step).
        DefaultStrategy is a no-op; IGS+ phase-2 (P1-5) uses this to schedule
        kornia Canny pre-passes on a subset of cameras."""

    @abstractmethod
    def state_dict(self) -> Dict[str, Any]:
        """Serializable per-strategy state for checkpoints. DefaultStrategy
        returns an empty dict (no state needed for bit-identical default
        behavior). MCMC includes its target_num_points."""

    @abstractmethod
    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        """Inverse of :meth:`state_dict`. DefaultStrategy is a no-op."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable strategy identifier used for cross-checkpoint validation.
        Per AGREEMENT §13 / §13.7, a mismatch between checkpoint's saved
        ``strategy.name`` and the current strategy's name must raise
        ``RuntimeError`` rather than silently loading."""

    def apply_permutation(self, perm: torch.Tensor) -> None:
        """Permute any per-Gaussian strategy-internal state along dim 0.

        P3-Morton (doc/specs/P3-Morton-spec.md §2.3): when the Gaussian
        tensors are reordered by Morton code, strategies that hold
        per-Gaussian state aligned with the slot index MUST also permute
        that state — otherwise IGS+ sampling scores (which are indexed
        by slot position in ``_free_mask`` / ``_error_score_max`` /
        ``_edge_score_cache``) end up applied to the wrong Gaussians.

        DefaultStrategy has no per-Gaussian state and inherits this
        no-op. IGS+ and MCMC override to permute their score buffers.

        Called by ``train.py`` *immediately after*
        ``gaussians.apply_permutation(perm)`` so the strategy's view of
        the world stays aligned with the GaussianModel.

        Args:
            perm: int64 tensor of shape ``[N]`` (the same permutation
                already applied to the GaussianModel tensors).
        """
        # Default: no strategy-internal per-Gaussian state. Concrete
        # strategies (IGS+, MCMC) override and permute their own buffers.
        return
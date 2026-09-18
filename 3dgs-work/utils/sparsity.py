"""P2-ADMM: consensus ADMM for Gaussian opacity sparsification.

LFS reference: ``LichtFeld-Studio/src/training/utils/sparsity.cuh`` (ADMM
pruning helper used to compact the Gaussian set after densification
plateaus). We port only the consensus-ADMM lite path: x is not modified
(no extra gradient step), only the auxiliary variables ``z`` and dual
``u`` are updated, and a final physical prune removes all Gaussians with
``z == 0`` (i.e. those that the proximal step decided are not worth
keeping).

Why opacity (and only opacity)?
    Opacity is the single per-Gaussian attribute whose "zero" state has
    an unambiguous physical meaning (invisible / dead Gaussian). Other
    attributes (xyz / scaling / rotation / SH) being zero would either
    be numerically degenerate (xyz=0 collapses onto origin) or have
    ambiguous semantics (SH=0 just means constant color, not "absent").
    This matches the LFS choice to ADMM-sparsify opacity before pruning.

Algorithm (per ADMMController.step):

    Given current ``opacity_i = sigmoid(_opacity[i])`` and running
    auxiliary ``z_i`` and dual ``u_i``:

        1. v = opacity - u                                          (residual)
        2. z = hard_threshold(v, tau)        where tau = sqrt(2*lambda/rho)
        3. u = u + rho * (z - opacity)

    After ``step`` converges (or at ``end_iter``), caller invokes
    ``controller.materialize_prune(gaussians)`` which builds a boolean
    mask ``z == 0`` and feeds it to ``GaussianModel.prune_points``.

This file deliberately has no CUDA dependencies — the per-Gaussian
operations are tiny (one subtraction, one compare, one update) and
PyTorch tensor ops on CUDA are already optimal at this scale.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class _ADMMConfig:
    """All ADMM knobs gathered for one controller."""

    rho: float
    lambda_: float
    start_iter: int
    step_every: int
    end_iter: int


class ADMMController:
    """Consensus-ADMM-lite controller for Gaussian opacity sparsity.

    Lifecycle::

        admm = ADMMController(opt)            # builds nothing yet
        for iteration in 1 .. N:
            if admm.engaged(iteration):
                admm.ensure_state(gaussians)  # lazy z/u init
                if iteration % step_every == 0:
                    admm.step(gaussians)      # proximal + dual update
        admm.materialize_prune(gaussians)     # physical prune at end

    Conventions:
        * All math is on the post-activation opacity (``get_opacity()``)
          — never on the raw ``_opacity`` logit. This keeps the proximal
          threshold in the same units as what the rasterizer sees.
        * Default values mirror LFS ``parameters.hpp`` defaults where
          documented; everything is 0/disabled in 3dgs-work until the
          user explicitly opts in.
    """

    def __init__(self, opt):
        # All flags default to 0 = disabled. Caller's responsibility
        # to instantiate only when ``opt.admm_rho > 0``.
        self.cfg = _ADMMConfig(
            rho=float(getattr(opt, "admm_rho", 0.0)),
            lambda_=float(getattr(opt, "admm_lambda", 0.0)),
            start_iter=int(getattr(opt, "admm_start_iter", 30_000)),
            step_every=int(getattr(opt, "admm_step_every", 100)),
            # end_iter == 0 means "use opt.iterations" — caller resolves
            # that lazily via ``resolve_end_iter``.
            end_iter=int(getattr(opt, "admm_end_iter", 0)),
        )
        self._z: torch.Tensor | None = None  # [N] — auxiliary var
        self._u: torch.Tensor | None = None  # [N] — dual var
        self._n_init: int = 0                # size when z/u was first allocated

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    def resolve_end_iter(self, opt_iterations: int) -> int:
        """``--admm_end_iter 0`` → fall back to ``opt.iterations``.

        Returns the absolute iteration at which ``materialize_prune``
        should run.
        """
        end = self.cfg.end_iter if self.cfg.end_iter > 0 else opt_iterations
        return max(end, self.cfg.start_iter)

    def engaged(self, iteration: int) -> bool:
        """Whether ADMM should be active at ``iteration``.

        Engages strictly at/after ``start_iter`` and only while rho > 0
        and lambda > 0. We don't gate on end_iter here — the caller
        decides when to invoke ``materialize_prune``.
        """
        if self.cfg.rho <= 0 or self.cfg.lambda_ <= 0:
            return False
        return iteration >= self.cfg.start_iter

    def ensure_state(self, gaussians) -> None:
        """Lazy init of z and u.

        z is initialised to the current opacity (so the first proximal
        step only zero-pulls the genuinely tiny entries, not all of
        them). u is initialised to zero. If the Gaussian set grew
        (densification happened between ADMM steps), z/u are padded
        with the new entries' opacity and zero respectively.
        """
        opacity = gaussians.get_opacity.detach().flatten()
        n = opacity.shape[0]
        if self._z is None:
            self._z = opacity.clone()
            self._u = torch.zeros_like(opacity)
            self._n_init = n
            return
        if n > self._z.shape[0]:
            pad = n - self._z.shape[0]
            self._z = torch.cat([self._z, opacity[-pad:].clone()])
            self._u = torch.cat([self._u, torch.zeros(pad, dtype=self._u.dtype, device=self._u.device)])

    # ------------------------------------------------------------------
    # Core ADMM step
    # ------------------------------------------------------------------

    def step(self, gaussians) -> None:
        """One consensus-ADMM-lite step: proximal + dual update.

        Proximal operator for ``lambda * |z|_0`` (L0 sparsity)::

            v = opacity - u
            z = v  if |v| > sqrt(2*lambda/rho)
                0  otherwise

        This is the canonical hard-thresholding rule for L0-ADMM
        (matching LFS ``sparsity.cuh`` ``S_threshold(v)``).
        """
        self.ensure_state(gaussians)
        opacity = gaussians.get_opacity.detach().flatten()
        # Threshold: |v| > sqrt(2*lambda/rho) keeps v, else 0.
        tau = (2.0 * self.cfg.lambda_ / self.cfg.rho) ** 0.5
        with torch.no_grad():
            v = opacity - self._u
            mask = v.abs() > tau
            self._z = torch.where(mask, v, torch.zeros_like(v))
            self._u = self._u + self.cfg.rho * (self._z - opacity)

    # ------------------------------------------------------------------
    # Inspection + materialize
    # ------------------------------------------------------------------

    @property
    def z(self) -> torch.Tensor:
        return self._z

    @property
    def u(self) -> torch.Tensor:
        return self._u

    def prune_mask(self) -> torch.Tensor:
        """Boolean mask ``[N]``: True where z is exactly 0 (prune target).

        Returns ``None`` if ADMM has not run yet (so the caller can
        short-circuit). Note this excludes ``get_opacity() < eps``
        (low-opacity but z > 0) — those are handled by the standard
        opacity-prune path in the strategy.
        """
        if self._z is None:
            return None
        return self._z == 0

    def materialize_prune(self, gaussians) -> int:
        """Apply ``prune_points(z == 0)`` and return the number pruned.

        After pruning, the z/u buffers are reset so the controller can
        be re-engaged later (not used in v1, but the API supports it).

        Mask shape contract: ``prune_points`` (and its private
        ``_prune_optimizer``) requires a 1-D ``[N]`` mask. Production
        callers in ``gaussian_model.py`` (e.g. ``densify_and_prune``)
        pass ``[N]`` masks — e.g. ``(self.get_opacity < min_opacity)
        .squeeze()``. We follow that contract here.
        """
        mask = self.prune_mask()
        if mask is None or mask.sum().item() == 0:
            return 0
        n_before = gaussians.get_xyz.shape[0]
        # Flatten to [N] — matches gaussian_model.py production callers.
        gaussians.prune_points(mask)
        self._z = None
        self._u = None
        self._n_init = 0
        return int(n_before - gaussians.get_xyz.shape[0])

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Snapshot of current ADMM state (for logging / TB)."""
        if self._z is None:
            return {"admm_engaged": False}
        return {
            "admm_engaged": True,
            "admm_z_nonzero_frac": float((self._z != 0).float().mean().item()),
            "admm_z_mean_abs": float(self._z.abs().mean().item()),
            "admm_u_norm": float(self._u.norm().item()),
            "admm_n": int(self._z.shape[0]),
            "admm_n_init": int(self._n_init),
        }
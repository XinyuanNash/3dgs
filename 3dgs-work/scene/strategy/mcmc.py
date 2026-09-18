"""MCMC densification strategy (P1-MultiStrategy, OPTIMIZATIONS.md §13.6).

Mirror of LFS ``mcmc.cpp`` (Kheradmand et al. 2024): instead of the
gradient-thresholded clone+split+prune dance used by the original 3DGS,
MCMC keeps a target Gaussian count and grows toward it by

    (a) live-clone: each refinement cycle, multinomially sample ``n_new``
        Gaussians from the live population with weights proportional to a
        per-Gaussian "underfit score" (proxy for LFS's pixel-error buffer);
        insert clones at the same xyz but with **fresh Adam state** so
        their optimizer trajectory starts clean.
    (b) dead-relocate: for every Gaussian whose opacity has fallen below
        ``mcmc_opacity_threshold`` (or whose rotation has degenerated),
        pick a live source via multinomial sampling, copy the live
        Gaussian's params + Adam state into the dead slot, then zero the
        live source's state so it can be dropped.

Plus: per-step xyz noise (LFS mcmc.cpp:639-677 inject_noise) scaled by
the current position-LR. This is what keeps MCMC from converging to
degenerate local minima.

Reference: LFS ``src/training/components/mcmc.cpp`` (read-only).
"""
from __future__ import annotations

import torch

from . import eq9
from .base import Strategy
from .factory import register


@register("mcmc")
class McmcStrategy(Strategy):
    """MCMC densification strategy — live-clone + dead-relocate + xyz noise.

    Constructor args are inherited from :class:`Strategy`:
        model:   GaussianModel (will mutate _xyz, params, Adam state).
        opt:     OptimizationParams (reads mcmc_* + max_cap + densify_*).
        dataset: ModelParams (unused; kept for interface uniformity).

    MCMC-specific knobs (all on ``opt``):
        mcmc_opacity_threshold : float  — Gaussians with sigmoid(opacity) <=
            this are "dead" and eligible for relocation.
        mcmc_min_cap           : int    — absolute minimum alive count
            (clamp on n_target); set to 0 to allow shrink-to-zero.
        mcmc_relocate_every    : int    — refine cadence in iterations
            (LFS default 100, per LFS src/core/parameters.hpp; matches
            3dgs's ``densification_interval=100``).
        mcmc_noise_lr          : float  — noise scale on xyz; multiplies
            the current position-LR (``xyz_scheduler_args(iter)``).
            LFS uses 5e5 against its own per-step ExponentialLR; 3dgs
            uses 1e5 calibrated against get_expon_lr_func() decay
            (see OPTIMIZATIONS.md §13.6 "scheduler-aware noise").
        max_cap                : int    — absolute maximum alive count.
            When > 0 and ``densification_strategy == "default"``, train.py
            auto-promotes to "mcmc" (see §13.7 train.py changes).
    """

    @property
    def name(self) -> str:
        return "mcmc"

    def apply_permutation(self, perm: torch.Tensor) -> None:
        """Permute MCMC per-Gaussian state to match Morton reorder.

        P3-Morton (doc/specs/P3-Morton-spec.md §2.3): MCMC holds a
        per-Gaussian ``_error_score_max`` buffer aligned with the slot
        index (lazily allocated by ``_ensure_error_score_max``). When
        ``GaussianModel.reorder_morton()`` permutes the Gaussians, this
        buffer must follow — otherwise the multinomial sampling for
        clone/relocate picks based on stale error scores belonging to
        different Gaussians.

        Slots beyond ``perm.numel()`` (newly grown since the last
        apply_permutation) keep their existing values (0.0 by convention).
        """
        n = int(perm.shape[0])
        buf = getattr(self, "_error_score_max", None)
        if buf is not None:
            if buf.numel() >= n:
                buf[:n] = buf[:n][perm]
            else:
                pad = torch.zeros(n - buf.numel(), dtype=buf.dtype, device=buf.device)
                self._error_score_max = torch.cat([buf, pad])
                self._error_score_max[:n] = self._error_score_max[:n][perm]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_refining(self, iteration: int) -> bool:
        """D.8 noise cadence (spec §5.3).

        LFS injects noise unconditionally during the refinement window.
        3dgs-work's position-LR scheduler decays to ~0 by 30k, so very-late
        iters carry a geometrically-zero multiplier anyway. We still want
        an explicit cadence so noise can't drift post-densify_until_iter
        under abnormal LR configurations — soft-cap at
        ``densify_until_iter + noise_tail_iters`` (default 5000).
        """
        opt = self.opt
        return iteration < opt.densify_until_iter + opt.noise_tail_iters

    def _ensure_error_score_max(self, n_total: int, device: torch.device) -> torch.Tensor:
        """Lazily allocate the LFS-style `_error_score_max` buffer.

        Mirrors LFS mcmc.cpp:140-156 — a [N] persistent tensor that stores
        the per-Gaussian pixel-error score as elementwise MAX over time
        (NOT cumulative SUM). Reset to zero every 2 densification windows
        (see ``_reset_error_score_window``). Bound the scale; this is what
        prevents 18K late-iter regression when using raw cumulative error
        as the sampling score.
        """
        buf = getattr(self, "_error_score_max", None)
        if buf is None or buf.numel() < n_total or buf.device != device:
            buf = torch.zeros(n_total, dtype=torch.float32, device=device)
            self._error_score_max = buf
            self._error_score_windows = 0
        return buf

    def _reset_error_score_window(self) -> None:
        """Zero the persistent `_error_score_max` and reset the window counter.

        Mirrors LFS mcmc.cpp:741-744: every 2 densification windows the
        `_error_score_max` is fully zeroed so the score distribution does
        not become permanently biased by historical high-error Gaussians.
        """
        buf = getattr(self, "_error_score_max", None)
        if buf is not None:
            buf.zero_()
        self._error_score_windows = 0

    @staticmethod
    def _n_total(model) -> int:
        """Total slot count (including dead). Mirror LFS
        ``_splat_data->size()``.
        """
        return int(model._xyz.shape[0])

    def _frozen_mask(self) -> torch.Tensor:
        """bool [N] True if Gaussian is frozen.

        Default returns all-False since the 3dgs-work GaussianModel has
        no freeze mechanism today (LFS has a richer data layer that
        supports per-Gaussian freezing; parity is opt-in via
        ``freeze_every > 0``, see spec §6.1). When freeze support lands
        in `gaussian_model.py`, the only place to wire it in is here --
        three callers (dead-mask, sampling weights, inject_noise) all
        read this single helper.
        """
        m = self.model
        return torch.zeros(
            m._xyz.shape[0], dtype=torch.bool, device=m._xyz.device,
        )

    def _compute_dead_mask(self) -> torch.Tensor:
        """bool [N] True if Gaussian is dead.

        Mirrors LFS ``strategy_utils.cpp:329-352``:
            op_dead := raw_opacity <= logit(min_opacity)
            rot_dead := ||q||^2 < 1e-8
        Returns the OR.

        Two semantic shifts vs the legacy ``_alive_mask``:
          1. RAW logit threshold (not post-sigmoid). ``min_opacity`` is
             post-sigmoid in LFS, so we convert via ``logit(p)`` for the
             comparison on the raw parameter side.
          2. SQUARED rotation norm (not linear). The squared form
             avoids a sqrt on the hot path and matches LFS exactly.

        Frozen Gaussians never count as dead (D.11).
        """
        import math
        m = self.model
        # RAW logit threshold. logit(p) = log(p / (1-p)).
        # Equivalent when comparing `_opacity_raw <= logit(min_opacity)`
        # against `_opacity_raw <= min_op_raw` for the binary alive/dead
        # partition. Compute fresh each call (cheap, self-documenting).
        min_op_raw = math.log(
            self.opt.mcmc_opacity_threshold
            / (1.0 - self.opt.mcmc_opacity_threshold)
        )
        opacity_raw = m._opacity.view(-1)                 # raw logit
        op_dead = opacity_raw <= min_op_raw

        # ‖q‖² < 1e-8 (squared norm; matches LFS pruning_kernels.cu:23).
        q = m._rotation                                    # (N, 4)
        q_norm_sq = (q * q).sum(dim=-1)                    # ‖q‖²
        rot_dead = q_norm_sq < 1e-8

        return (op_dead | rot_dead) & ~self._frozen_mask()

    def _get_sampling_weights(
        self,
        raw_score: torch.Tensor,
        live_indices: torch.Tensor,
    ) -> torch.Tensor:
        """C score-floor + frozen-row zeroing (spec §4.4).

        Applies a ``clamp_min(1e-12)`` floor on the live-slice score so
        the multinomial can never collapse to a single Gaussian (matches
        LFS mcmc.cpp:171 ``sample_weights = scores.clamp_min(1e-12f)``).

        Then zeros out any frozen row (D.11) so frozen Gaussians never
        get cloned or relocated. Frozen semantics aren't yet wired in
        the 3dgs-work GaussianModel — ``_frozen_mask`` returns all-False
        today, so this is a no-op for default runs.

        Returning a tensor of the SAME shape and dtype as ``raw_score``
        keeps the downstream ``torch.multinomial(score, ...)`` callers
        intact — this method is purely a numerical filter.
        """
        # Score floor (LFS mcmc.cpp:171 clamp_min(1e-12f)).
        score = raw_score.clamp_min(1e-12)
        # Frozen-row zeroing (D.11). Only rows in live_indices need the
        # gate — the score already only spans the live slice. Re-broadcast
        # ~frozen_mask[live_indices] back to the score shape.
        if score.numel() > 0:
            frozen_live = (~self._frozen_mask()[live_indices]).to(score.dtype)
            score = score * frozen_live
            # Re-apply the floor AFTER the multiply so a frozen Gaussian
            # that originally scored exactly zero doesn't trip the floor
            # upward into 1e-12 (which would resuscitate its sampling
            # weight). Order matters: zero FIRST, floor SECOND.
            score = score.clamp_min(1e-12)
        return score

    def _target_count(self, n_total: int) -> int:
        """Target alive count for the next refinement cycle.

        Mirrors LFS mcmc.cpp: ``int(1.05 * n_total)`` then clamped to
        ``[mcmc_min_cap, max_cap]``. **n_total** (slot count, including
        dead) rather than **n_alive** (D.6, mcmc.cpp:394-396) — LFS uses
        ``_splat_data->size()`` because it grows toward max_cap without
        physically pruning dead slots until later cycles.

        Spec §4.1: the truncate (``int()``) matters — LFS uses C++
        ``static_cast<int>`` which truncates toward zero.
        """
        opt = self.opt
        n_target = int(1.05 * n_total)
        if opt.max_cap > 0:
            n_target = min(n_target, opt.max_cap)
        n_target = max(n_target, int(opt.mcmc_min_cap))
        return n_target

    def _multinomial_sample(
        self,
        n_samples: int,
        scores: torch.Tensor,
        live_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Sample ``n_samples`` indices from ``live_indices`` with probability
        proportional to ``scores`` (already aligned to live_indices).

        Uses :func:`torch.multinomial` with replacement (LFS mcmc.cpp does
        the same — sampling without replacement would fail when n_samples
        > live population).
        """
        if n_samples == 0 or scores.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=scores.device)
        weights = scores.clamp_min(1e-6)
        sampled_local = torch.multinomial(weights, n_samples, replacement=True)
        return live_indices[sampled_local]

    def _histogram_counts(
        self,
        sampled_idxs: torch.Tensor,
        n_total: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Per-source occurrence count with LFS +1 baseline, clamped [1, n_max].

        Thin delegate to :func:`scene.strategy.eq9.histogram_counts`. See
        ``docs/P1-MCMC-LFS-parity-spec.md`` §3.1 (D.1/D.2/D.3/D.18) for
        the algorithm semantics and the spec-audit note on why the +1
        baseline is load-bearing.

        Added in [P1-MCMCParity-A] so test_mcmc_histogram.py can exercise
        the per-source count formula before Commit B wires it into
        clone/relocate callsites. No control flow change to existing
        methods.
        """
        return eq9.histogram_counts(
            sampled_idxs=sampled_idxs,
            n_total=n_total,
            device=device,
            n_max=int(getattr(self.opt, "mcmc_relocate_n_max", eq9.RELOCATION_N_MAX)),
        )

    def _ensure_binom_coefficients(self, device: torch.device) -> torch.Tensor:
        """Lazily allocate the Eq.(9) binomial-sign coefficient table.

        Mirrors LFS ``d_relocation_coefficients`` (mcmc_kernels.cu:34-35):
        51x51 float32, computed once at first use and re-uploaded if the
        device changes (e.g. when training resumes on a different GPU).
        Cheaper than re-computing on every clone/relocate cycle.
        """
        cached = getattr(self, "_binom_coeffs", None)
        if cached is None or cached.device != device:
            cached = eq9.precompute_coefficients().to(device)
            self._binom_coeffs = cached
        return cached

    def _apply_eq9_relocation(
        self,
        sampled_idxs: torch.Tensor,
        model,
    ) -> None:
        """Apply Eq.(9) — Kheradmand 2024 — to mutate source rows in place.

        Implements the 5-step Eq.(9) surgery per ``docs/P1-MCMC-LFS-parity-spec.md``
        §3.3 (D.1/D.2/D.3):

            Step 1 — already done by caller (multinomial sample).
            Step 2 — histogram: LFS +1 baseline, clamp [1, n_max=51].
            Step 3 — Eq.(9) compute new opacity_post + scale_post per source.
            Step 4 — write new values back to source rows in raw-logit /
                     log space (inverse-sigmoid of opacity, log of scale).
            Step 5 — caller proceeds with existing param-copy to clone /
                     relocate destination; the mutated source values flow
                     through automatically.

        Mirrors LFS ``mcmc_kernels.cu:60-105`` (relocation_kernel) +
        ``mcmc.cpp:265-272`` (clone histogram) + ``mcmc.cpp:455-461``
        (relocate histogram). The +1 baseline is load-bearing; see the
        spec's audit-trail note on the 2026-07-29 +1 bug.

        No-op when ``sampled_idxs`` is empty.
        """
        if sampled_idxs.numel() == 0:
            return
        device = sampled_idxs.device
        n_total = int(model._xyz.shape[0])

        # --- Step 2: per-source histogram with LFS +1 baseline ---
        ratios = self._histogram_counts(sampled_idxs, n_total, device)

        # --- Step 3: Eq.(9) compute new opacity + scale per source ---
        coeffs = self._ensure_binom_coefficients(device)
        src_op_post = torch.sigmoid(model._opacity.view(-1))[sampled_idxs]
        src_sc_post = torch.exp(model._scaling[sampled_idxs])
        new_op_post, new_sc_post = eq9.relocate_opacity_scale(
            src_op_post,
            src_sc_post,
            ratios,
            coeffs,
            min_opacity_post=self.opt.mcmc_opacity_threshold,
        )

        # --- Step 4: write mutated values back to source rows ---
        # Inverse-sigmoid the new opacity and log the new scale so the
        # values land correctly in the raw-logit/log parameter buffers.
        # Clamps protect against float32 under/overflow at the [0,1]
        # boundary; the post-mutation values are then automatically
        # copied to clone slots (Step 5a) or relocate slots (Step 5b)
        # by the existing param-copy code following this call.
        with torch.no_grad():
            new_op_raw = torch.logit(
                new_op_post.clamp(1e-7, 1.0 - 1e-7), eps=1e-7,
            )
            new_sc_raw = torch.log(new_sc_post.clamp(min=1e-10))
            model._opacity.view(-1)[sampled_idxs] = new_op_raw
            model._scaling[sampled_idxs] = new_sc_raw

    # ------------------------------------------------------------------
    # post_backward — the densification trigger (mirrors LFS mcmc.cpp)
    # ------------------------------------------------------------------

    def post_backward(
        self,
        iteration: int,
        viewspace_point_tensor,
        visibility_filter,
        radii,
        scene_extent: float,
        error_buffer=None,
    ) -> None:
        opt = self.opt
        m = self.model

        # Steps 1-3: bookkeeping identical to DefaultStrategy + MCMC's
        # opacity_visible_count accumulator.
        m.max_radii2D[visibility_filter] = torch.max(
            m.max_radii2D[visibility_filter], radii[visibility_filter]
        )
        m.add_densification_stats(viewspace_point_tensor, visibility_filter)
        m.opacity_visible_count[visibility_filter] += 1

        # Step 4: skip refine outside the densification window.
        if not self._is_refining(iteration):
            return

        # Steps 5-12: heavy lifting runs every ``mcmc_relocate_every`` iters.
        if iteration % opt.mcmc_relocate_every != 0:
            return

        # D.16 stop_refine clear (spec §6.5): on the LAST densification
        # window, zero the persistent `_error_score_max` and reset the
        # window counter so the post-densify tail (and any future re-engage)
        # starts from a clean slate. Matches LFS mcmc.cpp:741-744
        # (`_reset_error_score_window` is called when the current iter
        # crosses the densify_until_iter boundary).
        if iteration == opt.densify_until_iter:
            if hasattr(self, "_error_score_max") and self._error_score_max is not None:
                self._error_score_max.zero_()
            self._error_score_windows = 0

        # IMPORTANT: every index we use here MUST be in the OLD coord space
        # (i.e. < n_total). The viewspace grad buffer has n_total rows and
        # is NOT refreshed inside this function. ddensification_postfix
        # appends new slots at [n_total, n_total + n_clones), which are
        # outside the grad buffer; only the pre-clone indices are
        # viewspace-grad-safe.
        n_total = m.get_xyz.shape[0]
        alive_mask = ~self._compute_dead_mask()               # (N,) bool
        live_indices = torch.nonzero(alive_mask, as_tuple=False).squeeze(-1)
        dead_indices = torch.nonzero(~alive_mask, as_tuple=False).squeeze(-1)
        n_total_alive = int(live_indices.numel())

        if viewspace_point_tensor is None or viewspace_point_tensor.grad is None:
            # No gradient signal — skip rather than crash. Matches
            # DefaultStrategy's "no grad → no densify" behavior.
            return

        # Step 5: per-Gaussian "underfit score".
        #
        # P1-7+ (OPTIMIZATIONS.md §13.6): when --p1_7_plus is set and the
        # rasterizer filled the persistent error buffer this iteration,
        # use that as the score. The buffer holds accumulated L1 pixel
        # error per Gaussian (sum of alpha_g * T_after * pixel_err), so
        # high-error Gaussians are preferentially cloned/relocated into
        # under-rendered regions — matching LFS's _error_score_max
        # semantics. Falls back to the legacy grad_norm * opacity proxy
        # when the buffer is missing/empty (e.g. legacy MCMC run).
        use_p1_7_plus_score = (
            getattr(opt, "p1_7_plus", False)
            and error_buffer is not None
            and isinstance(error_buffer, torch.Tensor)
            and error_buffer.numel() == n_total
        )
        opacity_score = torch.sigmoid(m.get_opacity[live_indices].squeeze(-1))
        if use_p1_7_plus_score:
            # P1-7+ score (mirrors LFS mcmc.cpp:707-744 _error_score_max).
            #
            # The rasterizer's `error_buffer` is a per-iteration accumulator
            # (atomicAdd of alpha_g * T_after * pixel_err_p); it's cleared
            # by the next forward pass. Using the raw cumulative SUM as the
            # sampling score causes 18K late-iter regression on campus
            # (-0.81 dB vs baseline, see Phase 3b gate) because:
            #   (a) the per-Gaussian error grows without bound over 18K
            #       iterations,
            #   (b) a few outlier Gaussians dominate the multinomial
            #       sampling and get over-cloned,
            #   (c) late-iter convergence is disrupted by the cloning bias.
            #
            # LFS fixes this in two steps (mcmc.cpp:707 + 741-744):
            #   1. `_error_score_max = max(_error_score_max, info[1])`
            #      (elementwise MAX, not SUM — bounds per-Gaussian scale).
            #   2. Every 2 densification windows, zero `_error_score_max`
            #      entirely (prevents long-term drift).
            #
            # We replicate both here in pure PyTorch: the persistent
            # `_error_score_max` accumulates the elementwise max over
            # this-iter's `error_buffer`, and is reset every 2 windows.
            score_max = self._ensure_error_score_max(
                n_total, error_buffer.device,
            )
            # Elementwise MAX over the first n_total slots — past growth
            # has appended zeros (lazy_alloc), so the new slots get
            # max(0, new_err) = new_err. Alive coord space untouched.
            score_max[:n_total] = torch.maximum(
                score_max[:n_total], error_buffer.to(score_max.dtype),
            )
            # LFS windowed reset: every 2 densification windows, fully zero
            # the persistent score so the multinomial distribution cannot
            # drift over very long runs.
            self._error_score_windows = (
                getattr(self, "_error_score_windows", 0) + 1
            )
            if self._error_score_windows >= 2:
                self._reset_error_score_window()
            # Score = MAX-window-error * sigmoid(opacity). clamp_min(1e-6)
            # matches LFS sampling-weight floor (mcmc.cpp:171 uses 1e-12
            # but we use 1e-6 here so the multinomial can never collapse
            # to a single Gaussian on the first iteration).
            err_score = score_max[live_indices].clamp_min(1e-6)
            raw_score = (err_score * opacity_score)
        else:
            # Legacy proxy: viewspace grad-norm * sigmoid(opacity).
            grad_norm = torch.norm(
                viewspace_point_tensor.grad[live_indices, :2], dim=-1
            )                                                # (n_alive,)
            raw_score = (grad_norm * opacity_score)
        # C score-floor + frozen-row zeroing (spec §4.4): single helper so
        # the floor and the frozen-aware zero-out live in exactly one place.
        score = self._get_sampling_weights(raw_score, live_indices)
        # Same scoring reuse for both clone (sample live src) and relocate
        # (sample live src per dead). Computed once.
        # NOTE: LFS uses ``n_total`` (slot count, includes dead) not
        # ``n_alive`` for the target computation — see _target_count docstring
        # and mcmc.cpp:394-396. We pass the local n_total computed above.
        n_target = self._target_count(n_total)
        # MCMC grows TOWARD max_cap, not within current buffer (per §13.6
        # spec: target = clamp(int(1.05 × n_total), min_cap, max_cap)).
        # We always append the delta to the end of the buffer — even when
        # n_total_alive << n_total (lots of dead), because cloning requires new
        # slots at the end, not overwriting dead slots. The dead slots are
        # later RELOCATED below (Step 10) once they exist.
        n_new_total = max(0, n_target - n_total_alive)
        # Edge case: every Gaussian is dead. Nothing to clone from and no
        # source for relocate — bail. (LFS handles this by simply not
        # refining; preserve that behaviour.)
        if n_total_alive == 0:
            return

        # Step 10 (relocate) is computed FIRST while indices still align
        # with the viewspace grad buffer. The actual surgery is applied
        # below, after the clone step. Why: relocate writes to dead slots
        # (positions in the OLD coord space), and clone APPENDS rather
        # than replaces — so the two operations don't conflict on indices.
        src_for_relocate = None
        if dead_indices.numel() > 0 and live_indices.numel() > 0:
            sampled_local = torch.multinomial(
                score, dead_indices.numel(), replacement=True,
            )
            src_for_relocate = live_indices[sampled_local]

        # Step 8-9: live-clone. Multinomial sample n_new_total live srcs,
        # each becomes a new clone appended at index [n_total, n_total+n_new).
        if n_new_total > 0 and live_indices.numel() > 0:
            new_clones_src = self._multinomial_sample(
                n_new_total, score, live_indices
            )
            if new_clones_src.numel() > 0:
                # [P1-MCMCParity-B] D.1/D.2/D.3: mutate source rows via
                # Eq.(9) BEFORE the clone copy so the appended slots
                # pick up the new opacity + scale (spec §3.3 Step 4+5).
                self._apply_eq9_relocation(new_clones_src, m)

                n_clones = new_clones_src.numel()
                new_xyz           = m._xyz[new_clones_src]
                new_features_dc   = m._features_dc[new_clones_src]
                new_features_rest = m._features_rest[new_clones_src]
                new_opacities     = m._opacity[new_clones_src]
                new_scaling       = m._scaling[new_clones_src]
                new_rotation      = m._rotation[new_clones_src]
                new_tmp_radii     = (
                    m.tmp_radii[new_clones_src] if m.tmp_radii.numel() > 0
                    else torch.zeros(n_clones, device=new_xyz.device)
                )
                # clone_target_index_old = n_total (pre-clone size)
                m.densification_postfix(
                    new_xyz, new_features_dc, new_features_rest,
                    new_opacities, new_scaling, new_rotation, new_tmp_radii,
                )
                # Clone indices post-growth = [n_total_old, n_total_new).
                n_total_old = m.get_xyz.shape[0] - n_clones
                m.reset_state(
                    torch.arange(n_total_old, m.get_xyz.shape[0], device="cuda")
                )

        # Step 10 (apply relocate surgery using pre-clone paired indices).
        # NB: src_for_relocate and dead_indices were sampled / computed
        # BEFORE the clone step, so they reference positions in the
        # current buffer — the clone only APPENDED at the end.
        if src_for_relocate is not None and src_for_relocate.numel() > 0:
            n_post = m.get_xyz.shape[0]
            assert int(dead_indices.max()) < n_post, (
                f"dead index {int(dead_indices.max())} >= post-growth "
                f"model size {n_post}"
            )
            assert int(src_for_relocate.max()) < n_post, (
                f"relocate src index {int(src_for_relocate.max())} >= "
                f"post-growth model size {n_post}"
            )
            # [P1-MCMCParity-B] D.1/D.2/D.3: mutate source rows via
            # Eq.(9) BEFORE the relocate copy so the dead slots pick
            # up the new opacity + scale (spec §3.3 Step 4+5). Same
            # rationale as the clone path above.
            self._apply_eq9_relocation(src_for_relocate, m)

            # Copy param values src -> dead slot (in place).
            with torch.no_grad():
                m._xyz[dead_indices]            = m._xyz[src_for_relocate]
                m._features_dc[dead_indices]   = m._features_dc[src_for_relocate]
                m._features_rest[dead_indices] = m._features_rest[src_for_relocate]
                m._opacity[dead_indices]       = m._opacity[src_for_relocate]
                m._scaling[dead_indices]       = m._scaling[src_for_relocate]
                m._rotation[dead_indices]      = m._rotation[src_for_relocate]
                if m.tmp_radii.numel() == m.get_xyz.shape[0]:
                    m.tmp_radii[dead_indices]  = m.tmp_radii[src_for_relocate]
            # Copy Adam state src -> dead, then zero src.
            m.relocate_state(src_for_relocate, dead_indices)
            # After relocate, the relocated Gaussian's old slot still
            # occupies a stale param/position, but its Adam state is now
            # zeroed so subsequent steps treat it as fresh. We do NOT
            # physically prune the source Gaussian here — the next
            # refine cycle's dead-relocate (or future prune) will handle
            # it. This matches LFS convention where the source is only
            # dropped once its slot is later marked dead.
            # However: src_for_relocate indices are NOT in dead_indices,
            # so this Gaussian stays alive until its opacity falls below
            # threshold on its own. Acceptable; LFS does the same.
            m.opacity_visible_count[dead_indices] = 0

        # Step 11: periodic opacity reset (matches DefaultStrategy cadence).
        # === Path C P0 #1 (P1-MCMC-PathC, 2026-08-14 supervisor approval):
        # opt-in flag to disable periodic opacity reset during MCMC
        # densification. LFS MCMC has NO opacity reset (LFS MCMC path never
        # calls reset_opacity; reset exists only in ImprovedGSPlus::reset_opacity).
        # 3dgs calls m.reset_opacity() every opacity_reset_interval=3000 iters,
        # which clamps all opacities to <=0.01 via replace_tensor_to_optimizer
        # — destroying high-opacity Gaussians. Audit: doc/status/
        # STATUS_P1-MCMC-PathC-Audit.md Top-5 #1. Default OFF preserves
        # existing behavior (bit-identical when OFF).
        if (
            iteration % opt.opacity_reset_interval == 0
            and not getattr(opt, "mcmc_disable_opacity_reset", False)
        ):
            m.reset_opacity()

        # Step 12: zero out accumulators so the next cycle starts fresh.
        m.xyz_gradient_accum = torch.zeros(
            (m.get_xyz.shape[0], 1), device=m._xyz.device,
        )
        m.denom = torch.zeros((m.get_xyz.shape[0], 1), device=m._xyz.device)
        m.max_radii2D = torch.zeros((m.get_xyz.shape[0],), device=m._xyz.device)

    # ------------------------------------------------------------------
    # step — xyz noise injection (LFS mcmc.cpp:639-677 inject_noise)
    # ------------------------------------------------------------------

    def step(self, iteration: int) -> None:
        if not self._is_refining(iteration):
            return
        # === Path A opt-in (P1-MCMC-PathA, 2026-08-13 supervisor approval):
        # disable noise injection entirely when --mcmc_disable_noise is set.
        # Per Ablation E diagnostic, noise LR scale mismatch (3dgs LR 1.6e-4
        # vs LFS LR 1.6e-5) causes noise to destroy opacity Gaussians during
        # the plateau phase. Default OFF preserves existing behavior; users
        # opt in via --mcmc_disable_noise. Densification (clone/relocate) is
        # handled by post_backward and is NOT affected by this flag.
        if getattr(self.opt, "mcmc_disable_noise", False):
            return
        # === END Path A ===
        opt = self.opt
        m = self.model
        with torch.no_grad():
            # Scheduler-aware noise (per OPTIMIZATIONS.md §13.6, 2026-07-27
            # supervisor decision): mirror LFS mcmc.cpp:639-677 inject_noise
            # shape but use 3dgs's *position_lr* scheduler rather than LFS's
            # per-step ExponentialLR. The hardcoded noise scale is
            # ``mcmc_noise_lr`` (default 1e5, calibrated for 3dgs's
            # get_expon_lr_func() decay over 30K iters; LFS uses 5e5
            # against its own per-step decay — DO NOT copy 5e5).
            #
            # The LFS kernel scales the noise by per-Gaussian covariance
            # (cov @ randn) for an anisotropic jitter aligned with the
            # Gaussian's own shape. We replicate that here in pure PyTorch
            # so MCMC works on the stock 3DGS rasterizer without a custom
            # CUDA kernel. Without covariance scaling, a plain
            # "noise * lr * 1e5" jitter blows up the geometry
            # (multi-unit stddev on a 10-unit scene).
            lr = m.xyz_scheduler_args(iteration)
            scale = opt.mcmc_noise_lr
            if scale == 0.0 or lr == 0.0:
                return
            # Per-Gaussian covariance: cov = R @ diag(exp(2*s)) @ R^T
            scaling = m._scaling                                     # (N, 3)
            s2 = torch.exp(2.0 * scaling)                            # (N, 3) diag
            rot = m._rotation                                        # (N, 4)
            # D.10 quaternion normalize (spec §5.4).
            # LFS mcmc_kernels.cu:136-143 normalizes via ``q * rsqrt(dot)``
            # with an ``fminf(rsqrt, 1e12f)`` upper cap so a degenerate
            # q doesn't blow up the rotation matrix. We mirror that here
            # in pure PyTorch: divide by clamp(sqrt(dot), min=1e-6) which
            # is equivalent to ``q * rsqrt(dot)`` when dot > 1e-12.
            q_norm = torch.sqrt((rot * rot).sum(dim=-1, keepdim=True))
            q_norm = torch.clamp(q_norm, min=1e-6)
            rot = rot / q_norm
            # === Path C P0 #2 (P1-MCMC-PathC, 2026-08-14 supervisor approval):
            # opt-in flag to use LFS-aligned quaternion normalization. 3dgs
            # default uses `q / clamp(sqrt(dot), min=1e-6)` which DIVIDES by
            # tiny q_norm when dot < 1e-12 → HUGE normalized q → garbage R →
            # garbage cov → garbage noise. LFS uses `q * min(rsqrt(dot), 1e12)`
            # which MULTIPLIES by bounded reciprocal → bounded normalized q →
            # bounded noise. The audit estimates +0.5 to +2.0 dB at late iters
            # for data5 where many Gaussians have near-degenerate rotations.
            # Default OFF preserves 3dgs-original.
            if getattr(self.opt, "mcmc_lfs_quat_norm", False):
                q_dot = (rot * rot).sum(dim=-1, keepdim=True)
                # rsqrt(dot) clamped to max=1e12, mirror LFS exactly
                rsqrt = torch.rsqrt(torch.clamp(q_dot, min=1e-24))
                rsqrt = torch.clamp(rsqrt, max=1e12)
                rot = rot * rsqrt
            # Build R from quat (xyzw) — same convention as gaussian_model.py
            # r = q.w, x = q.x, y = q.y, z = q.z
            r, x, y, z = rot[:, 0], rot[:, 1], rot[:, 2], rot[:, 3]
            R = torch.zeros(rot.shape[0], 3, 3, device=rot.device)
            R[:, 0, 0] = 1.0 - 2.0 * (y*y + z*z)
            R[:, 0, 1] = 2.0 * (x*y - r*z)
            R[:, 0, 2] = 2.0 * (x*z + r*y)
            R[:, 1, 0] = 2.0 * (x*y + r*z)
            R[:, 1, 1] = 1.0 - 2.0 * (x*x + z*z)
            R[:, 1, 2] = 2.0 * (y*z - r*x)
            R[:, 2, 0] = 2.0 * (x*z - r*y)
            R[:, 2, 1] = 2.0 * (y*z + r*x)
            R[:, 2, 2] = 1.0 - 2.0 * (x*x + y*y)
            # cov = R @ diag(s2) @ R^T
            S2 = torch.diag_embed(s2)                                # (N,3,3) diag
            cov = R @ S2 @ R.transpose(1, 2)                        # (N,3,3)
            # Standard normal noise
            noise = torch.randn_like(m._xyz)                         # (N,3)
            # transformed_noise = cov @ noise
            transformed = torch.einsum("nij,nj->ni", cov, noise)
            # LFS op_sigmoid = sigmoid(100 * sigmoid(opacity) - 0.5) —
            # suppresses noise for high-confidence Gaussians, preserves
            # it for under-confident ones. Same shape as LFS kernel.
            opacity_sig = torch.sigmoid(m._opacity).squeeze(-1)     # (N,)
            op_sigmoid = torch.sigmoid(100.0 * opacity_sig - 0.5)    # (N,)
            # Final scale = lr * mcmc_noise_lr * op_sigmoid per Gaussian
            noise_factor = (lr * scale * op_sigmoid).unsqueeze(-1)   # (N,1)
            # D.11 frozen-row zeroing (spec §5.5): frozen Gaussians must
            # not receive inject_noise jitter. Multiply by ~frozen_mask
            # broadcast so frozen rows contribute exactly 0.0 noise.
            frozen = (~self._frozen_mask()).to(noise_factor.dtype).unsqueeze(-1)
            m._xyz.add_(noise_factor * transformed * frozen)

    # ------------------------------------------------------------------
    # Strategy ABC boilerplate
    # ------------------------------------------------------------------

    def is_refining(self, iteration: int) -> bool:
        return self._is_refining(iteration)

    def on_iteration_end(self, iteration: int, scene_extent: float, radii) -> None:
        # No-op for MCMC — clone/relocate happens in post_backward.
        pass

    def state_dict(self) -> dict:
        """MCMC's per-iteration state.

        Currently nothing beyond what the GaussianModel itself captures
        (the model's optimizer state + opacity_visible_count is in
        ``gaussians.capture()``). Return an empty dict to keep the
        state_dict contract uniform with DefaultStrategy.
        """
        return {}

    def load_state_dict(self, sd: dict) -> None:
        # No-op: see state_dict().
        pass
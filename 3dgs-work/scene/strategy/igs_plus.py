"""IGS+ Phase-1 + Phase-2 densification strategy.

Mirrors LFS ``improved_gs_plus.cpp`` post_backward + LAS_densify.
Phase-1 covers the LFS quadratic budget + free-slot reuse + fallback
logic (P1-4.5). Phase-2 adds an opt-in Canny edge score combine
(P1-5: ``--igs_plus_use_edge=True`` by default since 2026-08-04 per
supervisor decision; opt-out via ``--no_igs_plus_use_edge``) that augments
sampling weights toward Gaussians near image edges (matches LFS
``improved_gs_plus.cpp:266-381`` + ``:609-619``).

CLI opt-in only. No modifications to mcmc.py, defaults.py, eq9.py,
adam_state_ops.py, the rasterizer submodule, or train.py.

Decoupling invariant (per spec §7.1):
    $ git diff <a9fd2a2>..HEAD -- scene/strategy/mcmc.py defaults.py \
        eq9.py base.py factory.py adam_state_ops.py \
        submodules/diff-gaussian-rasterization train.py | wc -l
    must be 0 after this commit lands.

Free-slot reuse semantics (LFS-style):
  Pruning NEVER physically shrinks the model's parameter tensors (no
  ``model.prune_points`` call). Instead, "pruned" slots get their data
  zeroed (opacity/scaling raw → large-negative → sigmoid/exp ≈ 0; colors
  zeroed) so the rasterizer effectively skips them, and ``_free_mask[i] =
  False`` marks the slot as available for reuse. ``_free_mask.numel()``
  therefore monotonically grows (or stays the same) and tracks total
  slot count; ``_free_mask.sum()`` tracks the active count.

  Filling reuses free slots: the LAS child (post-split xyz/scale/opacity
  derived from a pre-mutation snapshot of the parent) is written into
  the slot, and the slot's stale Adam state is zeroed via
  ``m.reset_state`` (P1-5.6; LFS ``fill_free_slots_with_data:1005-1048``).
  Momentum is never inherited. ``_free_mask[dst] = True`` marks the
  slot active.

  Growing appends zero rows via ``GaussianModel.add_state_zeros(n)`` and
  extends ``_free_mask`` with ``n`` new True entries.

This deviates from the spec's literal §6 ``_prune_post_reset``/``_opacity_prune``
which call ``model.prune_points`` — that method physically shrinks tensors
and would break the invariant ``_free_mask.numel() >= model.get_xyz.shape[0]``
required for free-slot reuse. The LFS reference uses logical-only pruning.

P1-5.6 (Long-Axis Split) replaces the P1-4.5/P1-5.5 pure-clone path with
LFS ``LAS_densify`` semantics (improved_gs_plus.cpp:405-585 +
densification_kernels.cu:674-775): each sampled parent is split into two
mirror halves along its longest local axis, with parent scale halved on
the longest axis and ×0.85 on the others, opacity ×0.6 in logit space.
"""
import math

import torch

from .base import Strategy
from .factory import register
from utils.general_utils import build_rotation, inverse_sigmoid


# Parameter group names from GaussianModel._ADAM_PARAM_NAMES
_PARAM_NAMES = ("xyz", "f_dc", "f_rest", "opacity", "scaling", "rotation")

# Map param group name -> attribute name on GaussianModel
_PARAM_ATTR = {
    "xyz":      "_xyz",
    "f_dc":     "_features_dc",
    "f_rest":   "_features_rest",
    "opacity":  "_opacity",
    "scaling":  "_scaling",
    "rotation": "_rotation",
}


# === [P1-5.6] LAS constants — LFS densification_kernels.cu:723-726 ===
_LOG_05 = math.log(0.5)      # parent/child longest-axis scale shrink
_LOG_085 = math.log(0.85)    # parent/child other-axis scale shrink

# Opacity clamp epsilon for the LAS logit transform
# (LFS inverse_sigmoid(densification_kernels.cu:20-25) uses 1e-7, NOT the
# 1e-6 used by P0-LAS at scene/gaussian_model.py:617). Locked here so the
# bit-equivalence to LFS is preserved end-to-end.
_LAS_OPACITY_EPS = 1e-7


@register("igs_plus")
class IgsPlusStrategy(Strategy):
    """LFS IGS+ Phase-1 (no Canny edge filter, free-slot reuse enabled).

    Args:
        model:   :class:`scene.gaussian_model.GaussianModel` to operate on.
        opt:     :class:`arguments.OptimizationParams` — densification knobs
                 + IGS+-specific flags (``igs_plus_*``).
        dataset: Optional dataset reference (unused for phase-1).
    """

    def __init__(self, model, opt, dataset=None):
        super().__init__(model, opt, dataset)
        n_init = int(model.get_xyz.shape[0])
        device = model.get_xyz.device

        max_cap = int(getattr(opt, "igs_plus_max_cap", 1_000_000))

        # ---- §3.2 buffers ----
        # _free_mask: True = active slot, False = free slot (LFS convention
        # opposite; this matches GaussianModel._deleted_mask idiom and is
        # GPU-friendly for ``(~mask).nonzero()`` lookups).
        self._free_mask = torch.ones(n_init, dtype=torch.bool, device=device)

        # _budget_schedule: LFS quadratic schedule from n_init to max_cap
        # over (densify_until_iter - densify_from_iter) / densification_interval
        # refine cycles. Length _total_steps (1-based LFS indexing).
        self._budget_schedule = self._get_count_array(
            n_init, max_cap,
            int(opt.densify_from_iter),
            int(opt.densify_until_iter),
            int(opt.densification_interval),
        )

        # _error_score_max: per-Gaussian pixel-error max (P1-7+ integration).
        # Sized to current slot count; grown alongside _free_mask in _grow().
        self._error_score_max = torch.zeros(
            n_init, dtype=torch.float32, device=device,
        )

        # P1-5 phase-2 buffers.
        # _edge_score_cache: per-Gaussian Canny edge score averaged over
        # the sampled cameras in the last on_iteration_end Canny pass.
        # Sizing mirrors LFS improved_gs_plus.cpp:316-322 ensure_error_score_shape;
        # grown in _grow() to match new slot count.
        self._edge_score_cache = torch.zeros(
            n_init, dtype=torch.float32, device=device,
        )
        # _edge_cache_valid: True iff on_iteration_end ran a Canny pass
        # in the current refine window. Phase-2 caller treats this as the
        # "edge scores available" gate; if False, falls back to phase-1.
        self._edge_cache_valid = False
        # _cam_image_cache / _cam_w2v_cache: GPU-cached training camera
        # images + world_view_transform matrices. Computed once on first
        # Canny call (camera set is constant during training). Avoids
        # repeated CPU→GPU transfer of 9MB images per camera per refine
        # iter (the dominant P1-5 wall-clock cost in the naive impl).
        self._cam_image_cache = None  # type: ignore[assignment]
        self._cam_w2v_cache = None  # type: ignore[assignment]
        self._cam_cache_n_cams = -1
        self._cam_cache_dev = None  # type: ignore[assignment]
        # _cam_canny_cache: precomputed Canny binary edge map for each
        # training camera. Shape [n_cams, H, W] float32 in {0, 1}.
        # Since camera images are constant during training, the Canny
        # edge structure is constant too — only the per-Gaussian
        # projection (which depends on moving Gaussian centers) needs
        # to be re-run per refine iter. This avoids 88ms × 145 refine
        # iters = ~13s of redundant Canny compute.
        self._cam_canny_cache = None  # type: ignore[assignment]
        self._cam_canny_H = -1
        self._cam_canny_W = -1

        self._current_step = 0

    # --------------------------------------------------------------
    # §4.1 Budget schedule (LFS quadratic — improved_gs_plus.cpp:209-232)
    # --------------------------------------------------------------
    @staticmethod
    def _get_count_array(
        n_init: int,
        max_cap: int,
        start_refine: int,
        stop_refine: int,
        refine_every: int,
    ) -> torch.Tensor:
        """LFS improved_gs_plus.cpp:209-232 quadratic schedule.

        ``schedule[i] = a * i^2 + b * i + c`` for ``i in [1, _total_steps]``,
        where:
            _total_steps = (stop_refine - start_refine) // refine_every + 2
            slope_lower_bound = (max_cap - n_init) / _total_steps
            k = 2 * slope_lower_bound
            a = (max_cap - n_init - k * _total_steps) / (_total_steps ** 2)
            b = k
            c = n_init

        Returns ``[_total_steps]`` int64 schedule aligned with LFS index i.
        Index 0 is unused (LFS uses 1-based); callers should treat
        ``sched[i-1]`` as the budget for refine iteration i.
        """
        span = int(stop_refine) - int(start_refine)
        n_init = int(n_init)
        max_cap = int(max_cap)
        if span <= 0 or max_cap <= n_init:
            # No refine window or no growth needed: degenerate singleton.
            return torch.tensor([n_init], dtype=torch.int64)
        total_steps = span // int(refine_every) + 2
        slope_lower_bound = (max_cap - n_init) / total_steps
        k = 2.0 * slope_lower_bound
        a = (max_cap - n_init - k * total_steps) / (total_steps ** 2)
        b = k
        c = float(n_init)
        idx = torch.arange(1, total_steps + 1, dtype=torch.float64)
        sched = a * idx * idx + b * idx + c
        sched = sched.clamp(min=float(n_init), max=float(max_cap)).round().to(torch.int64)
        return sched

    # --------------------------------------------------------------
    # §3.5 Sampling weight (phase-1 score + phase-2 edge combine)
    # --------------------------------------------------------------
    def _compute_phase1_scores(self) -> torch.Tensor:
        """Phase-1 sampling score (P1-4.5 logic, extracted).

        P1-7+ path (``_error_score_max`` nonzero): use the buffer directly.
        Fallback (P1-7+ OFF or no errors yet): uniform over active slots.

        Returns ``[N_max]`` float32 tensor aligned with ``_free_mask``.
        Mirrors LFS ``improved_gs_plus.cpp:266-313`` minus the Canny loop.
        """
        n_active = int(self._free_mask.sum().item())
        if n_active == 0:
            return torch.zeros_like(self._error_score_max)

        if float(self._error_score_max.abs().sum().item()) > 0.0:
            # P1-7+ path: raw error scores (clamped to score floor)
            scores = self._error_score_max.clone().clamp_min(
                float(self.opt.igs_plus_score_floor)
            )
        else:
            # Fallback: uniform 1.0 over active slots, 0.0 elsewhere
            scores = self._free_mask.to(self._error_score_max.dtype)
        # Mask free slots to exactly 0 so they cannot be sampled even
        # after clamp_min lifts other values. (Spec §3.5 has clamp_min
        # before masking, which lifts 0s to score_floor — a bug. We
        # apply mask *after* clamp to keep free slots dead.)
        scores = scores * self._free_mask.to(scores.dtype)
        return scores

    def compute_sampling_scores(self) -> torch.Tensor:
        """Per-slot sampling weight for LAS_densify.

        Phase-1 base (P1-4.5) combined with phase-2 edge score (P1-5).
        Mirrors LFS ``improved_gs_plus.cpp:325-381`` (densify_with_score
        with edge_score_weight=0.25). When --igs_plus_use_edge is False
        or the edge cache is invalid, returns phase-1 only.
        """
        base = self._compute_phase1_scores()
        if not bool(getattr(self.opt, "igs_plus_use_edge", False)):
            return base
        if not self._edge_cache_valid:
            return base

        edge = self._edge_score_cache
        # Resize to current slot count (mirror LFS ensure_error_score_shape
        # at improved_gs_plus.cpp:316-322). Should be a no-op if _grow has
        # run, but defends against operating on a stale cache.
        cur_n = int(self._free_mask.numel())
        if edge.numel() < cur_n:
            new_edge = torch.zeros(cur_n, dtype=torch.float32, device=edge.device)
            new_edge[:edge.numel()] = edge
            self._edge_score_cache = edge = new_edge
        elif edge.numel() > cur_n:
            edge = edge[:cur_n].clone()

        if float(edge.abs().sum().item()) <= 0:
            return base

        # Mirror LFS normalized_by_positive_median (improved_gs_plus.cpp:303-308):
        # divide by the median of positive entries; if all zeros, returns 0.
        pos = edge[edge > 0]
        if pos.numel() == 0:
            return base
        edge_norm = edge / float(pos.median().item())

        # LFS combine formula (improved_gs_plus.cpp:365):
        #     sampling_scores = normalized_error * (normalized_edge * EDGE_SCORE_WEIGHT + 1.0)
        # Here ``base`` plays the role of normalized_error.
        weight = float(self.opt.igs_plus_edge_score_weight)
        combined = base * (edge_norm * weight + 1.0)
        combined = combined * self._free_mask.to(combined.dtype)
        return combined

    # --------------------------------------------------------------
    # §3.6 densify_with_score (with phase-2 fallback chain)
    # --------------------------------------------------------------
    def _densify_with_score(self, scores: torch.Tensor, budget: int) -> None:
        """Sample ``min(budget_for_alloc, selectable)`` live srcs and LAS-densify.

        Mirrors LFS ``improved_gs_plus.cpp:325-403`` with the
        ``candidate_mask`` slice (LFS pre-filters low-error slots) stripped
        (cubic ramp's `compute_gaussian_score` already normalizes; phase-1
        computes a single combined score; phase-2 inherits this).

        P1-5 fallback chain (LFS improved_gs_plus.cpp:368-381):
          1. ``scores`` (phase-1 base + edge combine when --igs_plus_use_edge)
          2. edge-only fallback when phase-1 is too sparse
          3. uniform active fallback as last resort
        """
        n_active = int(self._free_mask.sum().item())
        if n_active == 0 or budget <= 0:
            return

        budget_for_alloc = max(0, int(budget) - n_active)
        if budget_for_alloc <= 0:
            return

        # Zero out free slots from the scoring distribution so they cannot
        # be sampled.
        masked = scores.masked_fill(~self._free_mask, 0.0)
        masked = masked.clamp_min(float(self.opt.igs_plus_score_floor))
        total = float(masked.sum().item())
        if total <= 0:
            return

        # Phase-1 budget allocation (no n_active cap yet — that cap is
        # the multinomial safety and goes right before torch.multinomial
        # so the fallback chain can run with a higher n_select).
        n_select = min(budget_for_alloc, int(total))
        if n_select <= 0:
            return

        # P1-5 fallback chain (LFS improved_gs_plus.cpp:369-381):
        # when phase-1 can't fill the budget:
        #   - if edge is enabled and has nonzero entries → use edge-only
        #     (this is the FINAL answer; LFS does not double-fall-back).
        #   - else (edge empty or disabled) → uniform active.
        if n_select < budget_for_alloc and bool(
            getattr(self.opt, "igs_plus_use_edge", False)
        ):
            if self._edge_cache_valid and float(
                self._edge_score_cache.abs().sum().item()
            ) > 0:
                edge_only = self._edge_score_cache.clone() * self._free_mask.to(
                    self._edge_score_cache.dtype
                )
                edge_total = float(edge_only.sum().item())
                if edge_total > 0:
                    masked = edge_only.clamp_min(1e-12)
                    n_select = min(budget_for_alloc, int(edge_total))
                    if n_select <= 0:
                        return
                    # Edge fallback takes over (LFS pattern). Apply the
                    # n_active cap right before multinomial.
                    n_select = min(n_select, n_active)
                    if n_select <= 0:
                        return
                    sampled = torch.multinomial(
                        masked, n_select, replacement=False,
                    )
                    self._las_densify(sampled)
                    return

        if n_select < budget_for_alloc:
            # Last resort: uniform over active slots so we still fill the
            # budget even when error scores and edge scores are both sparse.
            masked = self._free_mask.to(masked.dtype).clamp_min(1e-12)
            n_select = min(budget_for_alloc, int(self._free_mask.sum().item()))
            if n_select <= 0:
                return

        # P1-4.5 multinomial safety: cap n_select at n_active (the actual
        # number of eligible source slots). Must be the LAST step before
        # torch.multinomial — applying it earlier would short-circuit the
        # fallback chain.
        n_select = min(n_select, n_active)
        if n_select <= 0:
            return

        sampled = torch.multinomial(masked, n_select, replacement=False)
        self._las_densify(sampled)

    # --------------------------------------------------------------
    # §4.2 LAS_densify: Long-Axis Split (LFS-aligned)
    # --------------------------------------------------------------
    def _las_densify(self, sampled_idxs: torch.Tensor) -> None:
        """Long-Axis Split. Mirrors LFS improved_gs_plus.cpp:405-585 +
        densification_kernels.cu:674-775.

        P1-5.6 replaces the P1-4.5/P1-5.5 pure-clone path. Each sampled
        parent is SPLIT, not cloned:

          longest  = argmax(exp(log_scale))
          offset   = R[:, longest] * exp(log_scale[longest]) * 0.5
          parent.xyz = xyz + offset ; child.xyz = xyz - offset
          {parent, child}.log_scale[longest] += log(0.5)
          {parent, child}.log_scale[others]  += log(0.85)
          {parent, child}.opacity = inverse_sigmoid(sigmoid(opacity) * 0.6)
          child.{rotation, f_dc, f_rest} = parent's (unchanged)

        The pure clone left parent scale untouched, so large ellipsoids
        were never broken up and kept growing (supervisor: "椭球半径变得很大").

        Documented divergences from LFS bit-for-bit:
          * ``build_rotation`` (general_utils.py:78-99) normalises q ->
            R/||q||; LFS ``quat_to_rotmat`` (densification_kernels.cu:33-51)
            does not. Offsets differ by ||q||^2, < 1e-3 in practice (q is
            unit-init, Adam drift is small). Deliberate: matches in-tree
            P0-LAS at scene/gaussian_model.py:622.
          * Opacity clamp is 1e-7 (LFS), not P0-LAS's 1e-6.
        """
        n_samples = int(sampled_idxs.numel())
        if n_samples == 0:
            return

        m = self.model

        # 1) Snapshot parents BEFORE any mutation. The LFS kernel loads
        #    pos/quat/scale/opacity into registers (:697-711) and derives
        #    BOTH halves from them, so the child must never be computed
        #    from an already-mutated parent. ``.clone()`` is mandatory.
        with torch.no_grad():
            sel_xyz   = m._xyz[sampled_idxs].clone()             # [n,3]
            sel_log_s = m._scaling[sampled_idxs].clone()         # [n,3]
            sel_rot   = m._rotation[sampled_idxs].clone()        # [n,4]
            sel_op    = m._opacity[sampled_idxs].clone()         # [n,1]
            sel_fdc   = m._features_dc[sampled_idxs].clone()     # [n,1,3]
            sel_frest = m._features_rest[sampled_idxs].clone()   # [n,15,3]

            # 2) LAS math (densification_kernels.cu:716-775).
            sel_scaling = torch.exp(sel_log_s)                   # [n,3]

            # torch.argmax returns the FIRST maximum, matching LFS
            # ``get_max_value_index`` (:659-672). ``exp()`` is monotonic
            # so argmax over exp(log_s) == argmax over log_s.
            longest = sel_scaling.argmax(dim=-1)                 # [n]
            li = longest.unsqueeze(-1)                           # [n,1]

            # Offset magnitude uses the ORIGINAL scale, not the new one.
            offset_mag = sel_scaling.gather(1, li).squeeze(-1) * 0.5  # [n]

            new_log_s = sel_log_s + _LOG_085
            new_log_s.scatter_(
                1, li,
                sel_log_s.gather(1, li) + _LOG_05,
            )

            raw_sig = torch.sigmoid(sel_op) * 0.6
            new_op = inverse_sigmoid(torch.clamp(
                raw_sig, min=_LAS_OPACITY_EPS, max=1.0 - _LAS_OPACITY_EPS))

            # Offset direction = COLUMN ``longest`` of R. LFS reads
            # R[longest], R[longest+3], R[longest+6] from a row-major 3x3
            # (:734-737) == column. NOTE: ``build_rotation`` always
            # allocates on ``cuda`` (general_utils.py:82) regardless of
            # input device, so the ``.to()`` is REQUIRED for CPU callers
            # (unit tests).
            R = build_rotation(sel_rot).to(sel_rot.device)       # [n,3,3]
            off_dir = R.gather(
                2, longest.view(-1, 1, 1).expand(-1, 3, 1)
            ).squeeze(-1)                                        # [n,3]
            offset = off_dir * offset_mag.unsqueeze(-1)          # [n,3]

            parent_xyz = sel_xyz + offset
            child_xyz  = sel_xyz - offset

            # 3) Parent in-place mutation (LFS :755-761). Rotation and
            #    SH are deliberately untouched. If sampled_idxs ever held
            #    duplicates the repeated ``index_put_`` is order-
            #    nondeterministic but IDEMPOTENT (same value from one
            #    snapshot). P1-5.5 also did not dedupe; behaviour
            #    preserved.
            m._xyz[sampled_idxs]     = parent_xyz
            m._scaling[sampled_idxs] = new_log_s
            m._opacity[sampled_idxs] = new_op

        # 4) Parent Adam reset (LFS :503-508). NOT ``relocate_state``: the
        #    parent is new geometry, so momentum is discarded, not handed
        #    to a child.
        m.reset_state(sampled_idxs)

        # 5) Parent .grad reset (LFS :490-500). Observable because
        #    ``train.py`` runs ``post_backward`` (:207) BEFORE
        #    ``optimizer.step()`` (:228).
        self._zero_grad_at(sampled_idxs)

        # 6) Place children: free slots first, then append.
        free_idx = (~self._free_mask).nonzero(as_tuple=False).squeeze(-1)
        n_free = int(free_idx.numel())
        n_fill = min(n_samples, n_free)
        n_append = n_samples - n_fill

        # ``child`` holds the post-split tensor values (parent_xyz already
        # written in step 3); each fill/append caller slices [lo:hi] to
        # match its parent source.
        child = {
            "_xyz":           child_xyz,
            "_features_dc":   sel_fdc,
            "_features_rest": sel_frest,
            "_opacity":       new_op,
            "_scaling":       new_log_s,
            "_rotation":      sel_rot,
        }

        if n_fill > 0:
            fill = free_idx[:n_fill]
            self._write_child(fill, child, 0, n_fill, sampled_idxs[:n_fill])
            self._free_mask[fill] = True
            # A recycled slot still carries the Adam state of whatever
            # Gaussian used to live there — LFS clears it in
            # ``fill_free_slots_with_data`` (:1005-1048). Must be explicit.
            m.reset_state(fill)
            self._zero_grad_at(fill)

        if n_append > 0:
            n_old = int(self._free_mask.numel())
            self._grow(n_append)
            new = torch.arange(
                n_old, n_old + n_append,
                dtype=torch.long, device=self._free_mask.device,
            )
            self._write_child(new, child, n_fill, n_samples,
                              sampled_idxs[n_fill:])
            self._free_mask[new] = True
            # No reset_state: ``add_state_zeros`` ->
            # ``cat_tensors_to_optimizer`` (gaussian_model.py:511-531)
            # already zero-extended exp_avg/exp_avg_sq, matching LFS
            # ``extend_state_for_new_params`` (:577-582). No
            # ``_zero_grad_at`` either: ``cat_tensors_to_optimizer``
            # rebuilds each param as a fresh ``nn.Parameter`` whose
            # ``.grad`` is None (which also makes the step-5 grad-zeroing
            # a no-op on grow cycles — the whole grad is dropped, a
            # superset of zeroing).

        # 7) Resize _error_score_max to match new total slot count.
        # The next post_backward zeroing pass will reset it again, so the
        # zero-init here is mainly for size correctness (writes beyond the
        # current size would fail at the next torch.maximum()).
        self._error_score_max = torch.zeros(
            self._free_mask.numel(),
            dtype=torch.float32,
            device=self._error_score_max.device,
        )

    # --------------------------------------------------------------
    # §4.3 Write LAS child into dst slots (param values only; Adam
    #       state is owned by the caller — fill resets via
    #       ``m.reset_state``, append inherits zeros from
    #       ``add_state_zeros``).
    # --------------------------------------------------------------
    def _write_child(self, dst: torch.Tensor, child: dict,
                     lo: int, hi: int,
                     src: torch.Tensor) -> None:
        """Write rows ``[lo:hi]`` of the precomputed LAS child into
        slots ``dst``.

        Replaces P1-5.5's ``_copy_params``. The child is NOT a clone:
        ``child`` already holds the post-split xyz / log-scaling /
        opacity (computed from a pre-mutation snapshot of the parent).

        ``src`` is the matching parent index tensor, needed only to
        preserve P1-5.5's ``max_radii2D`` inheritance (all other
        tracking buffers zero out). Keeping those identical isolates
        the P1-5.6 diff to the LAS math + Adam surgery.

        Adam state is deliberately NOT touched here — the caller owns
        it (``m.reset_state`` on fill, ``add_state_zeros`` on append),
        mirroring LFS.
        """
        m = self.model
        with torch.no_grad():
            for attr, val in child.items():
                getattr(m, attr)[dst] = val[lo:hi]
            for attr in (
                "tmp_radii",
                "xyz_gradient_accum",
                "denom",
                "max_radii2D",
                "opacity_visible_count",
            ):
                buf = getattr(m, attr, None)
                if buf is None:
                    continue
                if buf.shape[0] >= int(self._free_mask.numel()):
                    buf[dst] = 0 if attr != "max_radii2D" else buf[src].clone()

    # --------------------------------------------------------------
    # §4.4 Zero .grad at given indices across all 6 param tensors
    #       (LFS grad parity: improved_gs_plus.cpp:490-500)
    # --------------------------------------------------------------
    def _zero_grad_at(self, idxs: torch.Tensor) -> None:
        """Zero ``.grad`` at ``idxs`` across all 6 param tensors.

        LFS zeroes ``state->grad`` alongside the Adam moments, both
        for split parents (improved_gs_plus.cpp:490-500) and recycled
        free slots (:1030-1048). Observable in our port because
        ``train.py`` calls ``strategy.post_backward`` (:207) BEFORE
        ``optimizer.step()`` (:228).

        ``.grad`` is None whenever ``_grow`` has just rebuilt the
        params (``cat_tensors_to_optimizer`` makes a fresh
        ``nn.Parameter``), and before the first backward — hence the
        guard.
        """
        if int(idxs.numel()) == 0:
            return
        m = self.model
        with torch.no_grad():
            for name in _PARAM_NAMES:
                p = getattr(m, _PARAM_ATTR[name], None)
                if p is None or p.grad is None:
                    continue
                p.grad[idxs] = 0.0

    # --------------------------------------------------------------
    # §4.4 Grow (LFS fill_free_slots_with_data grow path)
    # --------------------------------------------------------------
    def _grow(self, n_extra: int) -> None:
        """Append ``n_extra`` zero rows to all model buffers + ``_free_mask``.

        Uses ``GaussianModel.add_state_zeros(n)`` which:
          - appends zero rows to all 6 param tensors (with zero Adam state)
          - grows ``opacity_visible_count`` and ``tmp_radii`` to match

        ``add_state_zeros`` does NOT grow ``max_radii2D`` /
        ``xyz_gradient_accum`` / ``denom`` (they're size-frozen at
        ``training_setup`` time and only resized by
        ``densification_postfix`` / ``prune_points``). IGS+ uses
        ``add_state_zeros`` but our ``post_backward`` reads
        ``max_radii2D[visibility_filter]``, so we must resize those
        buffers here. Without this fix, the model grows past the
        size of those buffers and the next training step's
        ``m.max_radii2D[visibility_filter] = torch.max(...)`` writes
        to OOB indices → CUDA ``IndexKernel`` assertion (block
        [19,0,0], thread [9,0,0]). Hit in production at iter 14800
        of an 18K run on 2026-07-30.
        """
        if n_extra <= 0:
            return
        m = self.model
        n_old = int(self._free_mask.numel())
        m.add_state_zeros(n_extra)
        # Resize the tracking buffers that add_state_zeros missed.
        # Match densification_postfix pattern (gaussian_model.py:550-552).
        device = self._free_mask.device
        with torch.no_grad():
            new_n = int(m.get_xyz.shape[0])
            if hasattr(m, "max_radii2D") and m.max_radii2D.shape[0] < new_n:
                m.max_radii2D = torch.zeros(new_n, device=device)
            if hasattr(m, "xyz_gradient_accum") and m.xyz_gradient_accum.shape[0] < new_n:
                m.xyz_gradient_accum = torch.zeros(new_n, 1, device=device)
            if hasattr(m, "denom") and m.denom.shape[0] < new_n:
                m.denom = torch.zeros(new_n, 1, device=device)
        # New slots start as ACTIVE (= True); the caller (_las_densify)
        # sets them after _write_child, but mark them True here defensively
        # so a partial failure mid-copy doesn't leave the buffer with
        # phantom-active slots pointing to uninitialised data.
        new = torch.ones(n_extra, dtype=torch.bool, device=device)
        self._free_mask = torch.cat([self._free_mask, new])
        # Resize phase-2 edge cache to match new slot count (mirrors LFS
        # ensure_error_score_shape improved_gs_plus.cpp:316-322). New slots
        # start at 0 which means they CANNOT be sampled until the next
        # on_iteration_end Canny pass renormalizes them.
        self._edge_score_cache = torch.cat([
            self._edge_score_cache,
            torch.zeros(n_extra, dtype=torch.float32, device=device),
        ])
        # New slots invalidate the edge cache: callers downstream compare
        # edge scores by Gaussian index, and the new slots have no
        # meaningful edge score until the next Canny pass.
        self._edge_cache_valid = False
        # Sanity check: _free_mask tracks model size.
        assert (
            int(self._free_mask.numel()) == int(m.get_xyz.shape[0])
        ), (
            f"IGS+ invariant broken: _free_mask.numel()={int(self._free_mask.numel())} "
            f"!= model.get_xyz.shape[0]={int(m.get_xyz.shape[0])}"
        )
        assert n_old + n_extra == int(self._free_mask.numel())

    # --------------------------------------------------------------
    # §6 reset / prune (LFS-style: logical-only, no physical prune)
    # --------------------------------------------------------------
    def _reset_opacity(self) -> None:
        """Reset opacity to small initial value (mirrors LFS :586-608).

        Delegates to ``GaussianModel.reset_opacity`` which rebuilds the
        ``_opacity`` Parameter + zeros its Adam state.
        """
        self.model.reset_opacity()

    def _prune_post_reset(self) -> None:
        """Prune after reset_opacity. Logical-only — zeros slot data and
        marks free; does NOT physically shrink tensors.
        """
        self._opacity_prune(0)

    def _opacity_prune(self, iteration: int) -> None:
        """Periodic opacity-driven prune (LFS improved_gs_plus.cpp:834-846).

        Active slots with sigmoid(opacity) below the threshold are marked
        free: their data is zeroed and ``_free_mask[i] = False``. The slot
        rows remain in the model tensor (never physically shrunk) so
        subsequent LAS_densify calls can reuse them.
        """
        m = self.model
        threshold = float(self.opt.igs_plus_prune_opacity_threshold)
        op = m.get_opacity.squeeze(-1)
        mask = op < threshold
        if not bool(mask.any().item()):
            return
        # Zero out the slots so the rasterizer effectively skips them.
        idxs = mask.nonzero(as_tuple=False).squeeze(-1)
        with torch.no_grad():
            # sigmoid(-10) ≈ 4.5e-5, exp(-10) ≈ 4.5e-5
            NEG_INF_LIKE = -10.0
            m._opacity[idxs] = NEG_INF_LIKE
            m._scaling[idxs] = NEG_INF_LIKE
            m._features_dc[idxs] = 0.0
            m._features_rest[idxs] = 0.0
        # Mark free.
        self._free_mask[idxs] = False
        # Reset Adam state at the freed indices so a future fill slot
        # doesn't inherit stale moments from the old Gaussian.
        m.reset_state(idxs)

    # --------------------------------------------------------------
    # §3.5 + §5.3 Strategy interface
    # --------------------------------------------------------------
    def post_backward(
        self,
        iteration: int,
        viewspace_point_tensor,
        visibility_filter,
        radii,
        scene_extent: float,
        error_buffer=None,
    ) -> None:
        """Densification trigger. Mirrors LFS post_backward + simplified
        sync version of pre_step (no async edge pre-passes in phase-1).
        """
        opt = self.opt
        m = self.model

        # Track visibility / radii like default + MCMC strategies.
        if radii is not None and visibility_filter is not None:
            m.max_radii2D[visibility_filter] = torch.max(
                m.max_radii2D[visibility_filter], radii[visibility_filter]
            )

        # SH degree promotion (every sh_degree_interval).
        sh_interval = int(getattr(opt, "igs_plus_sh_degree_interval", 1000))
        if sh_interval > 0 and iteration % sh_interval == 0:
            m.oneupSHdegree()

        # P1-7+ error_buffer ingestion (only when --p1_7_plus is set).
        if error_buffer is not None and isinstance(error_buffer, torch.Tensor):
            n_err = int(error_buffer.numel())
            n_buf = int(self._error_score_max.numel())
            if n_err > 0 and n_buf > 0:
                n = min(n_err, n_buf)
                self._error_score_max[:n] = torch.maximum(
                    self._error_score_max[:n],
                    error_buffer[:n].to(self._error_score_max.dtype),
                )

        if not self.is_refining(iteration):
            # Outside refine window: clear score buffer at the boundary.
            if iteration == opt.densify_until_iter:
                self._error_score_max.zero_()
            return

        # P1-5 phase-2: Canny pre-pass. Spec says on_iteration_end is the
        # abstract hook; train.py doesn't call it, so we dispatch here
        # (the strategy is otherwise self-contained — no train.py edits).
        self.on_iteration_end(iteration, scene_extent, radii)

        # Sample + LAS_densify toward budget. LFS uses 1-based indexing on
        # schedule, so the first refine step (current_step=0) looks up
        # schedule[0] (which corresponds to LFS i=1 = start_refine). The
        # schedule length is _total_steps, not n_steps+1 (LFS pads with a
        # trailing copy at i=_total_steps).
        scores = self.compute_sampling_scores()
        budget_idx = min(
            self._current_step,
            int(self._budget_schedule.numel()) - 1,
        )
        budget = int(self._budget_schedule[budget_idx].item())
        self._densify_with_score(scores, budget)
        self._current_step += 1

        # Periodic opacity prune.
        prune_every = int(getattr(opt, "igs_plus_opacity_prune_every", 100))
        if prune_every > 0 and iteration % prune_every == 0:
            self._opacity_prune(iteration)

        # Periodic opacity reset + post-reset prune.
        # P1-5.5: LFS `prune_post_reset` is `[[maybe_unused]]` (not called).
        # Remove the post-reset prune to align with LFS behavior.
        reset_every = int(getattr(opt, "igs_plus_reset_every", 3000))
        if (
            reset_every > 0
            and iteration % reset_every == 0
            and iteration < opt.densify_until_iter
            and iteration > 0
        ):
            self._reset_opacity()
            # P1-5.5: removed `self._prune_post_reset()` (LFS does not call it)

        # Reset error buffer at end of refine iteration (matches LFS
        # reset-every-2-windows semantics, simplified to per-iter for
        # phase-1).
        self._error_score_max.zero_()

    def step(self, iteration: int) -> None:
        """No-op (default strategy convention). LFS step() runs async edge
        pre-passes; phase-1 omits them entirely.
        """
        pass

    def is_refining(self, iteration: int) -> bool:
        return (
            iteration >= self.opt.densify_from_iter
            and iteration < self.opt.densify_until_iter
            and iteration % self.opt.densification_interval == 0
        )

    def on_iteration_end(self, iteration: int, scene_extent: float, radii) -> None:
        """Phase-2 Canny pre-pass (P1-5, opt-in).

        Mirrors LFS ``improved_gs_plus.cpp:609-619`` (pre_step) + the
        Canny loop at ``:266-313``. Runs once per refine iteration:
          1. Sample N cameras (max(N, 8% of dataset); LFS random_cam_indices).
          2. For each camera: kornia Canny on the image, project all
             Gaussians to pixel coords, sample the binary edge map at
             those pixel coords, accumulate into per-Gaussian edge scores.
          3. Average across cameras; store in ``_edge_score_cache``; set
             ``_edge_cache_valid = True``.

        Determinism (per spec §4.2): camera sample is seeded by
        ``iteration`` so the same input always picks the same cameras.

        Called by ``post_backward`` (this strategy does not modify
        train.py, so the base class hook is invoked at the start of
        post_backward rather than at the end of the training loop).
        """
        if not bool(getattr(self.opt, "igs_plus_use_edge", False)):
            self._edge_cache_valid = False
            return
        if self.dataset is None:
            self._edge_cache_valid = False
            return
        if not self.is_refining(iteration):
            self._edge_cache_valid = False
            return
        self._compute_edge_scores(iteration)

    def _compute_edge_scores(self, iteration: int) -> None:
        """Run Canny + per-Gaussian pixel sampling; store in ``_edge_score_cache``.

        Mirrors LFS ``improved_gs_plus.cpp:266-313`` (``compute_gaussian_score``)
        with two simplifications:
          - Canny via kornia (CPU/GPU) instead of custom CUDA + bilateral filter.
          - Per-Gaussian pixel sampling via nearest-neighbor (no bilinear
            interpolation; ``__ldg`` precision in LFS is overkill for the
            Gaussian-coverage signal we want).

        Performance (P1-5 wall-clock gate): per-refine-iter work is O(B*N)
        GPU ops where B = #sampled cameras (≤10) and N = Gaussian count.
        The naive per-camera Python loop added ~7× wall-clock overhead vs
        phase-1 in initial measurement. Key optimizations:
          1. Cache training-camera images + world_view_transform on GPU
             once (camera set is constant during training).
          2. **Precompute Canny once for ALL training cameras** (camera
             images don't change; per-Gaussian projection does, so we
             only re-project + gather per iter, not re-Canny).
          3. Batch Canny across all n_cams cameras in one kornia call.
          4. Batched projection: einsum over (B, N, 4) instead of per-camera.
          5. Direct flat-index sampling via ``binary_flat[iy*W + ix]`` to
             skip the bool-mask fancy-index round-trip.
        """
        import kornia.filters  # lazy import: keep cold-start fast

        opt = self.opt
        # ---- camera sample (LFS random_cam_indices improved_gs_plus.cpp:796-815) ----
        try:
            cams = self.dataset.getTrainCameras()
        except (AttributeError, TypeError):
            self._edge_cache_valid = False
            return
        n_cams = len(cams)
        if n_cams == 0:
            self._edge_cache_valid = False
            return
        n_min = int(opt.igs_plus_edge_n_cameras)
        n_ratio = max(n_min, int(0.08 * n_cams))
        n_sample = min(n_cams, n_ratio)

        # Deterministic per-iteration camera selection (spec §4.2).
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(iteration))
        perm = torch.randperm(n_cams, generator=gen).tolist()
        sel_idx = perm[:n_sample]

        # ---- accumulators ----
        device = self._free_mask.device

        # Refresh GPU camera cache + Canny cache if needed (one-shot).
        if (
            self._cam_image_cache is None
            or self._cam_cache_n_cams != n_cams
            or self._cam_cache_dev != device
        ):
            self._build_cam_cache(cams, device)
        w2vs = self._cam_w2v_cache[sel_idx]    # [B, 4, 4] GPU

        # Use precomputed Canny outputs (constant across iters; only
        # per-Gaussian projection depends on moving Gaussian positions).
        binaries = self._cam_canny_cache[sel_idx]  # [B, H, W] GPU
        B, H, W = int(binaries.shape[0]), int(self._cam_canny_H), int(self._cam_canny_W)

        # ---- Batched projection: per-camera FoV intrinsics ----
        # Camera convention (cameras.py:86-88): world_view_transform is
        # stored transposed; the actual W2V is world_view_transform.T.
        # Full projection: p_view = (W2V @ p_world_h)^T
        # → output[b, n, i] = sum_j w2v[b, i, j] * xyz_h[n, j]
        n_slots = int(self._free_mask.numel())
        xyz = self.model.get_xyz  # [N, 3]
        n_xyz = int(xyz.shape[0])
        if n_xyz != n_slots:
            # Defensive: free_mask and model can briefly desync during
            # resize; truncate to model size (matches _grow invariant).
            n_xyz = min(n_xyz, n_slots)
            xyz = xyz[:n_xyz]
        xyz_h = torch.cat(
            [xyz, torch.ones(n_xyz, 1, device=device, dtype=xyz.dtype)], dim=1,
        )  # [N, 4]
        # Batched matmul: [B, 4, 4] @ [4, N] → [B, 4, N] → permute → [B, N, 4]
        p_view = torch.matmul(w2vs, xyz_h.T).permute(0, 2, 1).contiguous()  # [B, N, 4]
        z = p_view[:, :, 2]  # [B, N]
        valid = z > 0.01  # [B, N]
        # Per-camera intrinsics (FoV-based): broadcast scalars to [B]
        sel_cams = [cams[i] for i in sel_idx]
        fx = torch.tensor(
            [W / (2.0 * math.tan(float(c.FoVx) / 2.0)) for c in sel_cams],
            device=device, dtype=z.dtype,
        ).view(B, 1)  # [B, 1]
        fy = torch.tensor(
            [H / (2.0 * math.tan(float(c.FoVy) / 2.0)) for c in sel_cams],
            device=device, dtype=z.dtype,
        ).view(B, 1)  # [B, 1]
        # Safe div: replace z=0 with 1 to avoid divide-by-zero, then mask out.
        z_safe = torch.where(z > 0.01, z, torch.ones_like(z))
        pix_x = fx * p_view[:, :, 0] / z_safe + W / 2.0  # [B, N]
        pix_y = fy * p_view[:, :, 1] / z_safe + H / 2.0  # [B, N]
        ix = pix_x.long().clamp(0, W - 1)  # [B, N]
        iy = pix_y.long().clamp(0, H - 1)  # [B, N]

        # ---- Flat-index sampling: avoid bool-mask fancy-index round-trip ----
        flat = binaries.view(B, H * W)  # [B, H*W]
        idx_flat = (iy * W + ix)  # [B, N]
        sampled = torch.gather(flat, 1, idx_flat)  # [B, N]
        sampled = sampled * valid.to(sampled.dtype)
        accum = sampled.sum(dim=0) / float(B)  # [N]

        # ---- store ----
        if self._edge_score_cache.numel() < n_slots:
            new_cache = torch.zeros(n_slots, dtype=torch.float32, device=device)
            old_n = int(self._edge_score_cache.numel())
            new_cache[:old_n] = self._edge_score_cache
            self._edge_score_cache = new_cache
        elif self._edge_score_cache.numel() > n_slots:
            self._edge_score_cache = self._edge_score_cache[:n_slots].clone()
        self._edge_score_cache[:n_xyz].copy_(accum)

    def _build_cam_cache(self, cams, device) -> None:
        """One-shot GPU upload of training cameras' images + W2V matrices
        + precomputed Canny binary edge maps.

        Camera set + image content are constant during training, so:
          - images: pre-stacked into [n_cams, 3, H, W] GPU
          - W2V matrices: pre-stacked into [n_cams, 4, 4] GPU
          - Canny outputs: pre-computed into [n_cams, H, W] GPU
            (avoids re-running 88ms Canny per refine iter × 145 iters)

        Per-iter work in ``_compute_edge_scores`` shrinks to: project
        current Gaussian positions + gather from precomputed edge maps.
        """
        import kornia.filters  # lazy import

        imgs = []
        w2vs = []
        for cam in cams:
            img = cam.original_image  # [3, H, W]
            if img.device != device:
                img = img.to(device, non_blocking=True)
            imgs.append(img)
            w2v = cam.world_view_transform  # [4, 4] (stored transposed)
            if w2v.device != device:
                w2v = w2v.to(device, non_blocking=True)
            w2vs.append(w2v.T)  # store as the actual W2V
        self._cam_image_cache = torch.stack(imgs, dim=0).contiguous()  # [B, 3, H, W]
        self._cam_w2v_cache = torch.stack(w2vs, dim=0).contiguous()    # [B, 4, 4]
        self._cam_cache_n_cams = len(cams)
        self._cam_cache_dev = device

        # Precompute Canny once. Threshold defaults match spec §4.2.
        low = float(self.opt.igs_plus_canny_low_threshold)
        high = float(self.opt.igs_plus_canny_high_threshold)
        _, binaries = kornia.filters.canny(
            self._cam_image_cache, low, high,
        )  # [n_cams, 1, H, W]
        self._cam_canny_cache = binaries.squeeze(1).contiguous()  # [n_cams, H, W]
        self._cam_canny_H = int(self._cam_canny_cache.shape[1])
        self._cam_canny_W = int(self._cam_canny_cache.shape[2])
        self._edge_cache_valid = True

    def state_dict(self) -> dict:
        return {
            "_current_step": int(self._current_step),
            "_free_mask": self._free_mask.detach().cpu(),
            "_budget_schedule": self._budget_schedule.detach().cpu(),
        }

    def load_state_dict(self, sd: dict) -> None:
        if sd is None:
            return
        if "_current_step" in sd:
            self._current_step = int(sd["_current_step"])
        if "_free_mask" in sd:
            self._free_mask = sd["_free_mask"].to(self._free_mask.device)
        if "_budget_schedule" in sd:
            self._budget_schedule = sd["_budget_schedule"].to(
                self._budget_schedule.device
            )

    @property
    def name(self) -> str:
        return "igs_plus"

    def apply_permutation(self, perm: torch.Tensor) -> None:
        """Permute IGS+ per-Gaussian state to match Morton reorder.

        P3-Morton (doc/specs/P3-Morton-spec.md §2.3 trap #2): IGS+
        keeps three per-Gaussian state tensors aligned with the slot
        index in ``_free_mask`` / ``_error_score_max`` /
        ``_edge_score_cache``. When ``GaussianModel.reorder_morton()``
        permutes the underlying Gaussian tensors (xyz, scaling, etc.),
        IGS+'s scores get applied to the WRONG Gaussians unless we
        permute them by the same ``perm``.

        Empirical: skipping this permute caused PSNR to collapse to
        ~10 dB at iter 4000 (vs ~18 dB baseline) on data5 30K ship-path
        — the clone/relocate sampler was picking Gaussians by stale
        scores and dragging the schedule off-distribution. Adding this
        method fixed it (Gate 5 PASS).

        Buffers permuted (slot-aligned):
            _free_mask       [N] bool  — True = active slot
            _error_score_max [N] f32   — per-Gaussian pixel-error max
            _edge_score_cache[N] f32   — per-Gaussian Canny edge score

        NOT permuted (derived/cached, recomputed on next iter):
            self._cam_*_cache, _edge_cache_valid flag (gets recomputed
            on the next Canny pass anyway).

        The slot count may have grown (densify) since the previous
        apply_permutation call — we permute only the first ``perm.numel()``
        slots and let any newly-appended slots stay at their current
        values (which are 0/False for new slots, matching LFS convention).
        """
        n = int(perm.shape[0])
        # _free_mask: align with slot order. Slots beyond perm.numel()
        # (newly grown) keep their existing values.
        if hasattr(self, "_free_mask") and self._free_mask is not None:
            if self._free_mask.numel() >= n:
                self._free_mask[:n] = self._free_mask[:n][perm]
            else:
                # buffer shorter than perm — extend first (shouldn't
                # happen in practice; densify grows the buffer).
                pad = torch.zeros(n - self._free_mask.numel(),
                                  dtype=self._free_mask.dtype,
                                  device=self._free_mask.device)
                self._free_mask = torch.cat([self._free_mask, pad])
                self._free_mask[:n] = self._free_mask[:n][perm]
        # _error_score_max: same shape contract
        if hasattr(self, "_error_score_max") and self._error_score_max is not None:
            if self._error_score_max.numel() >= n:
                self._error_score_max[:n] = self._error_score_max[:n][perm]
            else:
                pad = torch.zeros(n - self._error_score_max.numel(),
                                  dtype=self._error_score_max.dtype,
                                  device=self._error_score_max.device)
                self._error_score_max = torch.cat([self._error_score_max, pad])
                self._error_score_max[:n] = self._error_score_max[:n][perm]
        # _edge_score_cache: same shape contract
        if hasattr(self, "_edge_score_cache") and self._edge_score_cache is not None:
            if self._edge_score_cache.numel() >= n:
                self._edge_score_cache[:n] = self._edge_score_cache[:n][perm]
            else:
                pad = torch.zeros(n - self._edge_score_cache.numel(),
                                  dtype=self._edge_score_cache.dtype,
                                  device=self._edge_score_cache.device)
                self._edge_score_cache = torch.cat([self._edge_score_cache, pad])
                self._edge_score_cache[:n] = self._edge_score_cache[:n][perm]
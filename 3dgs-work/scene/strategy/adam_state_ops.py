"""Adam-state surgery primitives (P1-MultiStrategy, OPTIMIZATIONS.md §13.7).

These are low-level operations that mutate the Adam (``exp_avg``,
``exp_avg_sq``) state of a :class:`torch.optim.Optimizer` *in place*. They are
the building blocks used by densification strategies (MCMC in P1-3, IGS+ in
P1-4/P1-5) to manage Adam moments when Gaussians are cloned / split / pruned /
relocated.

The primitives take an :class:`torch.optim.Optimizer` directly (not a
:class:`scene.gaussian_model.GaussianModel`) so they can be unit-tested with
a synthetic 4-param fixture — see ``tests/test_adam_state_ops.py``.

Reference contract (per OPTIMIZATIONS.md §13.14 row 1264, P1-2 verification):
    - After ``reset_state_at_indices``: state[indices] == 0, other indices
      retain their values.
    - After ``relocate_state_at_indices(src, dst)``: state[dst] (pre-call)
      == state[dst] (post-call) == state[src] (pre-call); state[src] == 0.
    - After 100 ``optimizer.step()`` post-surgery: no NaN, gradients finite.

Why not methods on :class:`GaussianModel`?
    The GaussianModel already has ``_prune_optimizer`` / ``cat_tensors_to_optimizer``
    / ``replace_tensor_to_optimizer`` (legacy). This module factors the *index-level*
    Adam mutation out so it can be:
      1. Unit-tested without instantiating a full Scene + Camera + dataset.
      2. Reused by future strategies that operate on subsets of Gaussians
         (e.g. MCMC's per-iteration clone/relocate) without forcing every
         call site to re-implement the bookkeeping.
"""
from typing import Iterable

import torch


def _get_param_group(optimizer: torch.optim.Optimizer, name: str):
    """Return the param-group dict whose ``group['name'] == name``.

    Raises ``KeyError`` with a helpful message if not found.
    """
    for group in optimizer.param_groups:
        if group.get("name") == name:
            return group
    available = [g.get("name") for g in optimizer.param_groups]
    raise KeyError(
        f"optimizer has no param group named {name!r}; "
        f"available names: {available}"
    )


def _require_state(optimizer: torch.optim.Optimizer, param: torch.Tensor):
    """Return the Adam state dict for ``param``.

    Raises ``RuntimeError`` if the state has not been initialised yet (i.e.
    ``optimizer.step()`` has never been called). The Adam-state surgery
    primitives below all require initialised state — calling them on a
    never-stepped optimizer is almost certainly a bug.
    """
    state = optimizer.state.get(param, None)
    if state is None:
        raise RuntimeError(
            f"param shape={tuple(param.shape)} has no Adam state; "
            f"optimizer.step() must run at least once before surgery "
            f"(call safe_state + a dummy loss.backward() + optimizer.step() "
            f"to initialise state)"
        )
    return state


def _normalize_indices(indices, param_shape_0: int, op_name: str) -> torch.Tensor:
    """Coerce indices to a 1-D long CUDA tensor and bounds-check.

    Args:
        indices: 1-D tensor-like of integer indices.
        param_shape_0: size of the first dim of the underlying param tensor.
        op_name: name of the calling op (used in error messages).

    Returns:
        ``torch.Tensor`` of shape ``(K,)`` with dtype ``torch.int64``.

    Raises:
        IndexError: any index is out of ``[0, param_shape_0)``.
    """
    idx = torch.as_tensor(indices, dtype=torch.int64)
    if idx.ndim != 1:
        raise ValueError(
            f"{op_name}: indices must be 1-D, got shape {tuple(idx.shape)}"
        )
    if idx.numel() > 0:
        if int(idx.min()) < 0 or int(idx.max()) >= param_shape_0:
            raise IndexError(
                f"{op_name}: index out of bounds "
                f"(min={int(idx.min())}, max={int(idx.max())}, "
                f"allowed range [0, {param_shape_0}))"
            )
    return idx


def reset_state_at_indices(
    optimizer: torch.optim.Optimizer,
    param_name: str,
    indices: Iterable[int],
) -> None:
    """Zero out Adam state (``exp_avg``, ``exp_avg_sq``) at the given row indices
    of the named param group. The parameter's ``.data`` is **not** touched.

    Typical use: a strategy just *cloned* Gaussians at ``indices`` (or just
    *relocated* a live Gaussian to ``indices``); the existing Adam moments at
    those rows would either be a leftover (zeros — still fine, but expensive)
    or, in the relocation case, stale moments from the previous Gaussian that
    occupied that slot. Resetting forces the strategy to start Adam from a
    clean slate at those rows.

    Args:
        optimizer: the :class:`torch.optim.Optimizer` whose state we mutate.
        param_name: ``group['name']`` identifying the param group to mutate.
            The first (``len == 1``) param of that group is operated on.
        indices: 1-D integer tensor of row indices to reset. May be on any
            device; will be moved to ``state["exp_avg"].device`` before use.
            Empty tensor is a no-op.

    Returns:
        ``None``. The optimizer state is mutated in place.

    Raises:
        KeyError: ``param_name`` not in ``optimizer.param_groups``.
        RuntimeError: the param has no Adam state (optimizer never stepped).
        ValueError: ``indices`` is not 1-D.
        IndexError: any index is out of ``[0, param.shape[0])``.

    See also:
        :func:`relocate_state_at_indices` for the src→dst + zero-src variant.
    """
    group = _get_param_group(optimizer, param_name)
    if len(group["params"]) != 1:
        raise RuntimeError(
            f"reset_state_at_indices only supports single-param groups, "
            f"got {len(group['params'])} params in group {param_name!r}"
        )
    param = group["params"][0]
    state = _require_state(optimizer, param)
    idx = _normalize_indices(indices, param.shape[0], "reset_state_at_indices")
    if idx.numel() == 0:
        return
    # Move to the same device as the state (typical: both on CUDA, but tests
    # may run on CPU — be defensive).
    idx = idx.to(state["exp_avg"].device)
    state["exp_avg"][idx] = 0
    state["exp_avg_sq"][idx] = 0


def relocate_state_at_indices(
    optimizer: torch.optim.Optimizer,
    param_name: str,
    src_indices: Iterable[int],
    dst_indices: Iterable[int],
) -> None:
    """Copy Adam state from ``src_indices`` to ``dst_indices``, then zero out
    ``src_indices``. The parameter's ``.data`` is **not** touched.

    Typical use: MCMC dead-relocation. A dead Gaussian (low opacity, slot
    ``dst``) gets its Adam moments replaced by a live Gaussian's (slot
    ``src``); the live Gaussian is then removed from the model (its slot's
    Adam moments are zeroed so any subsequent surgery or step sees a clean
    state).

    Args:
        optimizer: the :class:`torch.optim.Optimizer` whose state we mutate.
        param_name: ``group['name']`` identifying the param group to mutate.
        src_indices: 1-D integer indices whose state will be **read** then
            **zeroed**. Same length as ``dst_indices``.
        dst_indices: 1-D integer indices whose state will be **overwritten**
            with ``src_indices``' state. Same length as ``src_indices``.

    Returns:
        ``None``. The optimizer state is mutated in place.

    Raises:
        KeyError: ``param_name`` not in ``optimizer.param_groups``.
        RuntimeError: the param has no Adam state (optimizer never stepped),
            or the group has more than one param.
        ValueError: ``src_indices`` and ``dst_indices`` differ in length or
            either is not 1-D.
        IndexError: any index is out of ``[0, param.shape[0])``.

    Implementation note:
        The dst-write happens *before* the src-zero so that overlapping
        src/dst indices (e.g. ``relocate_state_at_indices(opt, name,
        [3], [3])``) behave as a no-op rather than losing data. Verified by
        ``test_relocate_state_overlapping_indices``.
    """
    group = _get_param_group(optimizer, param_name)
    if len(group["params"]) != 1:
        raise RuntimeError(
            f"relocate_state_at_indices only supports single-param groups, "
            f"got {len(group['params'])} params in group {param_name!r}"
        )
    param = group["params"][0]
    state = _require_state(optimizer, param)
    src = _normalize_indices(src_indices, param.shape[0], "relocate_state_at_indices")
    dst = _normalize_indices(dst_indices, param.shape[0], "relocate_state_at_indices")
    if src.numel() != dst.numel():
        raise ValueError(
            f"relocate_state_at_indices: src_indices ({src.numel()}) and "
            f"dst_indices ({dst.numel()}) must have the same length"
        )
    if src.numel() == 0:
        return
    device = state["exp_avg"].device
    src = src.to(device)
    dst = dst.to(device)
    # 1. Copy src -> dst FIRST. If src and dst overlap, this preserves the
    #    pre-call state in the overlap region (which is then zeroed in step 2).
    state["exp_avg"][dst] = state["exp_avg"][src].clone()
    state["exp_avg_sq"][dst] = state["exp_avg_sq"][src].clone()
    # 2. Zero src.
    state["exp_avg"][src] = 0
    state["exp_avg_sq"][src] = 0
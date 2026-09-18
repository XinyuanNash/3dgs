"""P2-PPISP: 4-stage differentiable Image Signal Processor.

Port of LichtFeld-Studio's PPISP (CVPR 2025, NVIDIA) to pure PyTorch.
Reference: LichtFeld-Studio/src/training/components/ppisp.hpp +
kernels/ppisp_math.cuh.

Four learnable stages, each a nn.Module:
  1. Exposure: per-frame scalar in log-space -> multiplicative gain (2^raw).
  2. Vignetting: per-camera per-channel 5-coeff polynomial on r^2/r^4/r^6.
  3. Color correction: per-frame 8-dim latent offsets -> 3x3 homography H,
     applied to (R, G, intensity) with intensity preservation.
  4. CRF (Camera Response Function): per-camera per-channel 4-param
     piecewise-power curve (toe, shoulder, gamma, center).

All stages support identity init (i.e., when parameters are zero / at their
identity values, the output equals the input). This is essential for
opt-in behavior: passing `--use_ppisp` with identity-initialized weights
must be bit-identical to no PPISP at all (during the pre-engage window).

Layout convention: apply() takes an image in HWC layout, returns HWC layout
(matches BilateralGrid convention). Pixel coordinates are (x=col, y=row).

Usage:
    ppisp = PPISP(num_frames=N_FRAMES, num_cameras=N_CAMERAS,
                  total_iterations=30000, start_iter=26000)
    out_hwc = ppisp(rgb_hwc, frame_idx=cam.uid(), camera_idx=0)
    loss.backward()  # autograd populates .grad on all 4 parameter tensors
    if ppisp.engaged(iteration):
        ppisp.optimizer_step()    # Adam + LR schedule
    ppisp.scheduler_step()        # always advances (so warmup tracks iters)
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Constants (mirror ppisp_math.cuh)
# ============================================================

PPISP_MAX_SHAPE_RAW = 16.0
# 2026-08-05: tightened from 1e-4 (LFS default). 1e-4 lets x/center reach
# 10000 in the piecewise-power CRF, then pow(10000, gamma) with gamma
# softplus-clamped at 16+ overflows to inf -> NaN at iter ~1200 of Adam
# stepping (Gate C 2026-08-05 reproducer). 0.05 caps x/center at 20, and
# pow(20, 16) = exp(16 * 3) ~ 2.6e6, well within fp32 range. The wider
# clamp means the piecewise-power "switch" is fuzzier (no longer
# bit-equivalent to LFS at identity), but the trade-off is stability.
PPISP_CENTER_EPSILON = 0.05
# 2026-08-05: NEW. Minimum argument to pow() in CRF forward. PyTorch
# autograd through pow(0, toe) computes d/dtoe [0^toe] = 0 * log(0) = NaN.
# Clamping to PPISP_POW_EPS keeps forward bit-identical (eps^toe vs 0^toe
# differ by < 1e-20) and eliminates NaN gradients. Mirrors what LFS avoids
# by hand-writing its backward (instead of relying on autodiff).
PPISP_POW_EPS = 1.0e-20
PPISP_MIN_EXPOSURE_EV = -16.0
PPISP_MAX_EXPOSURE_EV = 16.0

# ZCA pinv blocks for color correction (constant memory in LFS).
# Each block is a 2x2 matrix acting on (b, r, g, n) latent 2D offsets.
# Layout: [block_idx][row*2 + col].
PPISP_COLOR_PINV_BLOCKS = torch.tensor(
    [
        [0.0480542, -0.0043631, -0.0043631, 0.0481283],
        [0.0580570, -0.0179872, -0.0179872, 0.0431061],
        [0.0433336, -0.0180537, -0.0180537, 0.0580500],
        [0.0128369, -0.0034654, -0.0034654, 0.0128158],
    ],
    dtype=torch.float32,
)


# ============================================================
# Small math helpers (autograd-safe)
# ============================================================

def _finite_or_zero(x: torch.Tensor) -> torch.Tensor:
    """Replace non-finite (NaN/Inf) entries with 0 (matches LFS)."""
    return torch.where(torch.isfinite(x), x, torch.zeros_like(x))


def _sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable sigmoid (matches ppisp_sigmoid)."""
    x = _finite_or_zero(x)
    return torch.sigmoid(x)


def _exposure_value(raw: torch.Tensor) -> torch.Tensor:
    """Clamp exposure param to [MIN_EV, MAX_EV], non-finite -> 0."""
    return torch.clamp(_finite_or_zero(raw), PPISP_MIN_EXPOSURE_EV, PPISP_MAX_EXPOSURE_EV)


def _bounded_positive_forward(raw: torch.Tensor, min_value: float = 0.1) -> torch.Tensor:
    """Softplus-style bounded positive (matches ppisp_bounded_positive_forward).

    Used for CRF toe/shoulder/gamma. Output is always >= min_value.
    """
    raw = torch.clamp(_finite_or_zero(raw), max=PPISP_MAX_SHAPE_RAW)
    # min_value + softplus(raw) = min_value + log(1 + exp(raw))
    # LFS uses: min_value + max(raw, 0) + log(1 + exp(-|raw|))
    return min_value + F.softplus(raw)


def _inv_bounded_positive_forward(target: float, min_value: float) -> float:
    """Inverse of ``_bounded_positive_forward``: given a desired output value,
    return the raw input that produces it.

    bounded_positive_forward(raw) = min_value + softplus(raw) = target
    softplus(raw) = target - min_value
    raw = log(exp(target - min_value) - 1)

    Used for identity init of CRF params.
    """
    delta = target - min_value
    if delta <= 0:
        # Below the min_value; softplus can't reach it. Use a large negative.
        return -10.0
    return math.log(math.exp(delta) - 1.0)


def _clamped_forward(raw: torch.Tensor) -> torch.Tensor:
    """Sigmoid with center epsilon clamp (matches ppisp_clamped_forward)."""
    return torch.clamp(
        _sigmoid(raw),
        min=PPISP_CENTER_EPSILON,
        max=1.0 - PPISP_CENTER_EPSILON,
    )


# ============================================================
# Stage 1: Exposure (per-frame, log-space scalar)
# ============================================================

class Exposure(nn.Module):
    """Per-frame exposure scalar. Identity at raw=0 (factor = 2^0 = 1)."""

    def __init__(self, num_frames: int):
        super().__init__()
        self.num_frames = num_frames
        # Identity init: 0 -> factor 1.
        self.raw = nn.Parameter(torch.zeros(num_frames))

    def forward(self, rgb_hwc: torch.Tensor, frame_idx: int) -> torch.Tensor:
        # rgb_hwc: [H, W, 3]; output same shape.
        exposure_param = self.raw[frame_idx]
        factor = torch.exp2(_exposure_value(exposure_param))  # scalar in [2^-16, 2^16]
        return rgb_hwc * factor


# ============================================================
# Stage 2: Vignetting (per-camera per-channel, 5 coeffs)
# ============================================================

class Vignetting(nn.Module):
    """Per-camera per-channel polynomial vignetting.

    Params layout: [num_cameras, 3, 5] = (cx, cy, alpha0, alpha1, alpha2).
    Identity at all-zeros -> falloff = 1 everywhere.
    """

    def __init__(self, num_cameras: int):
        super().__init__()
        self.num_cameras = num_cameras
        # Identity init: cx=cy=0, alpha0..2=0 -> falloff = 1.
        self.raw = nn.Parameter(torch.zeros(num_cameras, 3, 5))

    def forward(
        self, rgb_hwc: torch.Tensor, camera_idx: int, H: int, W: int
    ) -> torch.Tensor:
        # rgb_hwc: [H, W, 3]; output same shape.
        # Vignetting params for this camera: [3, 5].
        params = _finite_or_zero(self.raw[camera_idx])  # [3, 5]
        max_res = float(max(H, W))
        # uv in centered normalized coords, range [-0.5, 0.5].
        ys = (torch.arange(H, device=rgb_hwc.device, dtype=rgb_hwc.dtype) - H * 0.5) / max_res
        xs = (torch.arange(W, device=rgb_hwc.device, dtype=rgb_hwc.dtype) - W * 0.5) / max_res
        u, v = torch.meshgrid(xs, ys, indexing="xy")  # both [H, W]
        # dx = u - cx, dy = v - cy, per channel.
        cx = params[:, 0].view(3, 1, 1)  # [3, 1, 1]
        cy = params[:, 1].view(3, 1, 1)
        a0 = params[:, 2].view(3, 1, 1)
        a1 = params[:, 3].view(3, 1, 1)
        a2 = params[:, 4].view(3, 1, 1)
        dx = u.unsqueeze(0) - cx  # [3, H, W]
        dy = v.unsqueeze(0) - cy
        r2 = dx * dx + dy * dy
        r4 = r2 * r2
        r6 = r4 * r2
        falloff = torch.clamp(1.0 + a0 * r2 + a1 * r4 + a2 * r6, min=0.0, max=1.0)
        # falloff: [3, H, W]; multiply rgb_hwc (H, W, 3) channel-wise.
        return rgb_hwc * falloff.permute(1, 2, 0)  # [H, W, 3]


# ============================================================
# Stage 3: Color correction (per-frame, 8-dim latent -> 3x3 H)
# ============================================================

class ColorCorrection(nn.Module):
    """Per-frame color correction via 3x3 homography.

    Params layout: [num_frames, 8] = (b_x, b_y, r_x, r_y, g_x, g_y, n_x, n_y).
    Identity at all-zeros -> homography H = identity (no color shift).

    The 8 latent offsets parameterize 4 chromaticity control points (Blue,
    Red, Green, neutral White). The homography is constructed by:
      1. Applying a fixed ZCA pinv block to each control point's offset.
      2. Forming target chromaticities t_b, t_r, t_g, t_gray.
      3. Building skew-symmetric matrix from t_gray and stacking with T.
      4. Recovering the line at infinity lambda_v via cross products.
      5. Constructing H = T * diag(lambda_v) * S_inv, then normalizing H[2,2]=1.
    """

    def __init__(self, num_frames: int):
        super().__init__()
        self.num_frames = num_frames
        # Identity init: all 8 dims zero -> homography = identity.
        self.raw = nn.Parameter(torch.zeros(num_frames, 8))
        # Register the constant ZCA pinv blocks (frozen).
        # Shape: [4, 2, 2] so PPISP_COLOR_PINV_BLOCKS[b] is the 2x2 block.
        pinv = PPISP_COLOR_PINV_BLOCKS.view(4, 2, 2)
        self.register_buffer("pinv_blocks", pinv)

    def _compute_homography(self, params_8: torch.Tensor) -> torch.Tensor:
        """Compute 3x3 homography from 8-dim latent. params_8: [..., 8]."""
        # Split into 4 (x, y) offsets.
        b = params_8[..., 0:2]  # [B, 2]
        r = params_8[..., 2:4]
        g = params_8[..., 4:6]
        n = params_8[..., 6:8]
        # Apply ZCA pinv blocks: each is a 2x2 matmul.
        # broadcast self.pinv_blocks [4, 2, 2] over batch.
        # Reshape for batched matmul: params as [B, 4, 2], pinv as [1, 4, 2, 2].
        stacked = torch.stack([b, r, g, n], dim=-2)  # [B, 4, 2]
        # Apply: out[b_idx, k] = sum_l pinv[k, l] * stacked[b_idx, l]
        # Equivalent: out = stacked @ pinv.transpose(-1, -2) over the last 2 dims.
        # pinv_blocks: [4, 2, 2] -> transpose to [4, 2, 2] (already row-major 2x2).
        out = torch.einsum("bki,kij->bkj", stacked, self.pinv_blocks)  # [B, 4, 2]
        bd = out[:, 0]  # [B, 2]
        rd = out[:, 1]
        gd = out[:, 2]
        nd = out[:, 3]
        # Build target chromaticities (3D homogeneous coords).
        t_b = torch.stack([bd[:, 0], bd[:, 1], torch.ones_like(bd[:, 0])], dim=-1)  # [B, 3]
        t_r = torch.stack([torch.ones_like(rd[:, 0]) + rd[:, 0], rd[:, 1], torch.ones_like(rd[:, 0])], dim=-1)
        t_g = torch.stack([gd[:, 0], torch.ones_like(gd[:, 1]) + gd[:, 1], torch.ones_like(gd[:, 0])], dim=-1)
        t_gray = torch.stack(
            [torch.full_like(nd[:, 0], 1.0 / 3.0) + nd[:, 0],
             torch.full_like(nd[:, 0], 1.0 / 3.0) + nd[:, 1],
             torch.ones_like(nd[:, 0])],
            dim=-1,
        )
        # T matrix: rows are (t_b.x, t_r.x, t_g.x), (t_b.y, t_r.y, t_g.y),
        # (t_b.z, t_r.z, t_g.z). Shape: [B, 3, 3]. We stack along dim=-1 so
        # T[i, j, 0] = t_b[i, j], T[i, j, 1] = t_r[i, j], T[i, j, 2] = t_g[i, j]
        # — i.e., T[i] row j is the j-th component of (t_b, t_r, t_g).
        # (dim=-2 would stack along the second axis, putting each full target
        # vector as a row instead — the wrong convention for this construction.)
        T = torch.stack([t_b, t_r, t_g], dim=-1)  # [B, 3, 3]
        # Skew-symmetric from t_gray: [0, -z, y; z, 0, -x; -y, x, 0].
        skew = torch.stack(
            [
                torch.stack([torch.zeros_like(t_gray[:, 0]), -t_gray[:, 2], t_gray[:, 1]], dim=-1),
                torch.stack([t_gray[:, 2], torch.zeros_like(t_gray[:, 0]), -t_gray[:, 0]], dim=-1),
                torch.stack([-t_gray[:, 1], t_gray[:, 0], torch.zeros_like(t_gray[:, 0])], dim=-1),
            ],
            dim=-2,
        )  # [B, 3, 3]
        # M = skew @ T.
        M = torch.matmul(skew, T)  # [B, 3, 3]
        # lambda_v via cross of row pairs; pick the first non-degenerate.
        r0 = M[:, 0]  # [B, 3]
        r1 = M[:, 1]
        r2 = M[:, 2]
        lambda_v = torch.cross(r0, r1, dim=-1)  # [B, 3]
        n2 = (lambda_v * lambda_v).sum(dim=-1, keepdim=True)  # [B, 1]
        # Fallback to r0 x r2 if degenerate.
        fallback1 = torch.cross(r0, r2, dim=-1)
        n2_fb = (fallback1 * fallback1).sum(dim=-1, keepdim=True)
        use_fallback1 = (n2 < 1.0e-20).float()
        lambda_v = lambda_v * (1.0 - use_fallback1) + fallback1 * use_fallback1
        n2 = n2 * (1.0 - use_fallback1) + n2_fb * use_fallback1
        # Fallback to r1 x r2 if still degenerate.
        fallback2 = torch.cross(r1, r2, dim=-1)
        use_fallback2 = (n2 < 1.0e-20).float()
        lambda_v = lambda_v * (1.0 - use_fallback2) + fallback2 * use_fallback2
        # S_inv = [[-1,-1,1],[1,0,0],[0,1,0]].
        S_inv = torch.tensor(
            [[-1.0, -1.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            device=T.device, dtype=T.dtype,
        )
        # D = diag(lambda_v).
        D = torch.diag_embed(lambda_v)  # [B, 3, 3]
        # H = T @ D @ S_inv.
        H = torch.matmul(torch.matmul(T, D), S_inv)
        # Normalize so H[2, 2] = 1.
        s = H[:, 2, 2]  # [B]
        # Avoid division by near-zero s.
        scale = torch.where(s.abs() > 1.0e-20, 1.0 / s, torch.ones_like(s))
        H = H * scale.view(-1, 1, 1)
        return H  # [B, 3, 3]

    def forward(self, rgb_hwc: torch.Tensor, frame_idx: int) -> torch.Tensor:
        # rgb_hwc: [H, W, 3]. Compute homography for this frame, apply.
        params_8 = self.raw[frame_idx].unsqueeze(0)  # [1, 8]
        H_mat = self._compute_homography(params_8)[0]  # [3, 3]
        # Apply H to (R, G, intensity) where intensity = R+G+B.
        R = rgb_hwc[..., 0]
        G = rgb_hwc[..., 1]
        B = rgb_hwc[..., 2]
        intensity = R + G + B  # [H, W]
        # rgi_in: [H, W, 3] = (R, G, intensity).
        rgi_in = torch.stack([R, G, intensity], dim=-1)
        # rgi_out = rgi_in @ H^T (since H is applied to row vectors on the right
        # in the LFS code: rgi_out[i, j] = sum_k H[k, j] * rgi_in[i, k]).
        # Equivalent to: rgi_out = rgi_in @ H.T.
        rgi_out = torch.matmul(rgi_in, H_mat.t())  # [H, W, 3]
        # Normalize so intensity is preserved: scale factor = intensity / rgi_out.z.
        norm_factor = intensity / (rgi_out[..., 2] + 1.0e-5)  # [H, W]
        rgi_out = rgi_out * norm_factor.unsqueeze(-1)
        # Reconstruct RGB: rgb_out = (rgi_out.x, rgi_out.y, rgi_out.z - x - y).
        rgb_out = torch.stack(
            [rgi_out[..., 0], rgi_out[..., 1], rgi_out[..., 2] - rgi_out[..., 0] - rgi_out[..., 1]],
            dim=-1,
        )
        return rgb_out  # [H, W, 3]


# ============================================================
# Stage 4: CRF (per-camera per-channel, 4 coeffs)
# ============================================================

class CRF(nn.Module):
    """Per-camera per-channel piecewise-power tone curve.

    Params layout: [num_cameras, 3, 4] = (toe, shoulder, gamma, center).
    Identity: toe=1, shoulder=1, gamma=1, center=0.5 -> linear (y = x).

    Piecewise-power CRF (matches ppisp_apply_crf in ppisp_math.cuh):
      toe, shoulder, gamma bounded by _bounded_positive_forward (>= min_value).
      center = sigmoid(raw_center) clamped to (eps, 1-eps) in (0, 1).
      lerp_val = (shoulder - toe) * center + toe
      a = shoulder * center / lerp_val
      b = 1 - a
      if x <= center:  y = a * (x / center) ** toe
      else:             y = 1 - b * ((1 - x) / (1 - center)) ** shoulder
      out = max(y, 0) ** gamma
    """

    def __init__(self, num_cameras: int):
        super().__init__()
        self.num_cameras = num_cameras
        # Identity init: pick raw values so bounded_positive_forward(raw) == 1.0
        # (for toe/shoulder/gamma) and clamped_forward(raw) == 0.5 (for center).
        # This makes the piecewise-power curve the identity y = x when params
        # are at init — critical for the opt-in invariant: frozen PPISP must
        # be bit-identical to no PPISP.
        toe_raw = _inv_bounded_positive_forward(1.0, 0.3)       # ~ 0.0136
        shoulder_raw = _inv_bounded_positive_forward(1.0, 0.3) # ~ 0.0136
        gamma_raw = _inv_bounded_positive_forward(1.0, 0.1)    # ~ 0.8969
        center_raw = 0.0                                       # sigmoid(0) = 0.5
        init = torch.tensor(
            [[toe_raw, shoulder_raw, gamma_raw, center_raw]],
            dtype=torch.float32,
        )
        self.raw = nn.Parameter(init.view(1, 1, 4).expand(num_cameras, 3, 4).clone())

    def forward(self, rgb_hwc: torch.Tensor, camera_idx: int) -> torch.Tensor:
        # rgb_hwc: [H, W, 3]; clamp to [0, 1] before CRF.
        rgb_clamped = torch.clamp(rgb_hwc, min=0.0, max=1.0)
        params = self.raw[camera_idx]  # [3, 4]
        # Vectorize: rgb_clamped permute to [3, H, W] so we can broadcast toe/
        # shoulder/gamma/center (each [3]) to [3, H, W] without a Python loop.
        x = rgb_clamped.permute(2, 0, 1)  # [3, H, W]
        toe = _bounded_positive_forward(params[:, 0], min_value=0.3).view(3, 1, 1)
        shoulder = _bounded_positive_forward(params[:, 1], min_value=0.3).view(3, 1, 1)
        gamma = _bounded_positive_forward(params[:, 2], min_value=0.1).view(3, 1, 1)
        center = _clamped_forward(params[:, 3]).view(3, 1, 1)
        lerp_val = (shoulder - toe) * center + toe
        a = (shoulder * center) / lerp_val
        b = 1.0 - a
        # x / center and (1 - x) / (1 - center) per-channel.
        x_lower = x / center
        x_upper = (1.0 - x) / (1.0 - center)
        # Clamp away from 0 BEFORE pow: pow(0, toe) gives 0 * log(0) = NaN
        # gradient through the exponent. Clamping to _POW_EPS keeps forward
        # bit-identical (0^toe vs eps^toe differ by < 1e-20) and kills the
        # NaN backward. 2026-08-05 Gate C: this single line moves the
        # NaN onset from iter 1213 to >3000 iters.
        x_lower_safe = torch.clamp(x_lower, min=PPISP_POW_EPS, max=1.0)
        x_upper_safe = torch.clamp(x_upper, min=PPISP_POW_EPS, max=1.0)
        y_lower = a * torch.pow(x_lower_safe, toe)
        y_upper = 1.0 - b * torch.pow(x_upper_safe, shoulder)
        y = torch.where(x <= center, y_lower, y_upper)
        # Same pow(0, gamma) gradient safety on the output.
        out = torch.pow(torch.clamp(y, min=PPISP_POW_EPS), gamma)
        return out.permute(1, 2, 0)  # back to [H, W, 3]


# ============================================================
# PPISP container (4 stages + Adam + reg loss)
# ============================================================

class PPISP(nn.Module):
    """Physically-Plausible Image Signal Processing for 3DGS.

    Holds 4 nn.Module stages + an Adam optimizer + LR scheduler.
    Implements `engaged(iter)` gate (per P0-Bilateral-v2 freeze-during-
    densification pattern), reg_loss, and opt-in default.

    Args:
        num_frames: number of training frames (per-frame params).
        num_cameras: number of unique cameras (per-camera params).
        total_iterations: total training iterations (for LR scheduler).
        start_iter: iteration at which the Adam step engages. Until then,
            gradients accumulate but no Adam step happens (freeze gate).
        lr: Adam learning rate (default 2e-3, matches LFS).
        reg_weight_exposure_mean: smooth_l1(mean(exposure), beta=0.1) weight.
        reg_weight_vig_center: mean(cx^2 + cy^2) weight.
        reg_weight_vig_channel: var across RGB channels of vig params.
        reg_weight_vig_non_pos: mean(relu(alpha0..alpha2)) weight.
        reg_weight_color_mean: smooth_l1(mean(color @ pinv), beta=0.005) weight.
        reg_weight_crf_channel: var across RGB channels of CRF params.
        beta1, beta2, eps: Adam hyperparams (LFS defaults).
        warmup_steps: linear warmup steps (factor warmup_start_factor -> 1.0).
        warmup_start_factor: initial LR multiplier during warmup.
        final_lr_factor: final LR multiplier at end of training (cosine).
    """

    def __init__(
        self,
        num_frames: int,
        num_cameras: int,
        total_iterations: int,
        start_iter: int = 26000,
        lr: float = 2e-3,
        reg_weight_exposure_mean: float = 1.0,
        reg_weight_vig_center: float = 0.02,
        reg_weight_vig_channel: float = 0.1,
        reg_weight_vig_non_pos: float = 0.01,
        reg_weight_color_mean: float = 1.0,
        reg_weight_crf_channel: float = 0.1,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1.0e-15,
        warmup_steps: int = 500,
        warmup_start_factor: float = 0.01,
        final_lr_factor: float = 0.01,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.num_cameras = num_cameras
        self.total_iterations = total_iterations
        self.start_iter = start_iter
        self.lr = lr
        # 4 stages.
        self.exposure = Exposure(num_frames)
        self.vignetting = Vignetting(num_cameras)
        self.color = ColorCorrection(num_frames)
        self.crf = CRF(num_cameras)
        # Reg weights.
        self.reg_weight_exposure_mean = reg_weight_exposure_mean
        self.reg_weight_vig_center = reg_weight_vig_center
        self.reg_weight_vig_channel = reg_weight_vig_channel
        self.reg_weight_vig_non_pos = reg_weight_vig_non_pos
        self.reg_weight_color_mean = reg_weight_color_mean
        self.reg_weight_crf_channel = reg_weight_crf_channel
        # Adam + LR scheduler.
        # We use a manual LR schedule (matches BilateralGrid pattern):
        # scheduler_step() runs every iter to update self._current_lr;
        # optimizer_step() applies gradients with self._current_lr (and is
        # gated by PPISP.engaged(iteration) in train.py).
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=lr,  # placeholder; self._current_lr is the actual LR used per-step
            betas=(beta1, beta2),
            eps=eps,
        )
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.warmup_steps = warmup_steps
        self.warmup_start_factor = warmup_start_factor
        self.final_lr_factor = final_lr_factor
        self.initial_lr = lr
        self._current_lr = lr * warmup_start_factor
        self._step = 0  # number of scheduler_step calls (NOT optimizer steps)
        self._opt_step = 0  # number of optimizer_step calls (for bias correction)

    def _lr_lambda(self, step: int) -> float:
        """Linear warmup -> cosine decay to final_lr_factor.

        step is the number of optimizer steps performed (NOT scheduler calls).
        Matches the LFS gsplat PPISP scheduler shape.
        """
        if step < self.warmup_steps:
            # Linear ramp: warmup_start_factor -> 1.0.
            f = step / max(1, self.warmup_steps)
            return self.warmup_start_factor + (1.0 - self.warmup_start_factor) * f
        # Cosine decay over remaining (total_iterations - warmup_steps) steps.
        progress = (step - self.warmup_steps) / max(
            1, self.total_iterations - self.warmup_steps
        )
        progress = min(progress, 1.0)
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.final_lr_factor + (1.0 - self.final_lr_factor) * cos

    def engaged(self, iteration: int) -> bool:
        """True iff the Adam step should run at `iteration`."""
        return int(iteration) >= self.start_iter

    def forward(
        self,
        rgb_hwc: torch.Tensor,
        frame_idx: int,
        camera_idx: int = 0,
    ) -> torch.Tensor:
        """Apply 4-stage ISP. rgb_hwc: [H, W, 3]; returns [H, W, 3].

        frame_idx is per-frame (used by Exposure, Color).
        camera_idx is per-camera (used by Vignetting, CRF).
        """
        H, W, _ = rgb_hwc.shape
        # Stage 1: exposure.
        x = self.exposure(rgb_hwc, frame_idx)
        # Stage 2: vignetting.
        x = self.vignetting(x, camera_idx, H, W)
        # Stage 3: color correction.
        x = self.color(x, frame_idx)
        # Stage 4: CRF.
        x = self.crf(x, camera_idx)
        return x

    # ------------------------------------------------------------
    # Regularization (matches LFS ppisp.cpp reg_loss_gpu)
    # ------------------------------------------------------------

    def reg_loss(self) -> torch.Tensor:
        """Total regularization loss. Returns a scalar GPU tensor.

        All terms are autograd-friendly so they accumulate gradients
        into the same parameter .grad tensors as the photometric loss.
        """
        loss = torch.zeros((), device=next(self.parameters()).device)
        # 1. Exposure mean: smooth_l1(mean(raw), beta=0.1).
        if self.reg_weight_exposure_mean > 0.0:
            exp_mean = self.exposure.raw.mean()
            # smooth_l1(x, beta): 0.5 * (x/beta)^2 if |x| < beta else |x| - 0.5*beta.
            beta = 0.1
            abs_x = exp_mean.abs()
            smooth = torch.where(
                abs_x < beta,
                0.5 * (exp_mean / beta) ** 2,
                abs_x - 0.5 * beta,
            )
            loss = loss + self.reg_weight_exposure_mean * smooth
        # 2. Vig center: mean(cx^2 + cy^2).
        if self.reg_weight_vig_center > 0.0:
            cx = self.vignetting.raw[..., 0]
            cy = self.vignetting.raw[..., 1]
            vig_center = (cx * cx + cy * cy).mean()
            loss = loss + self.reg_weight_vig_center * vig_center
        # 3. Vig non-pos: mean(relu(alpha)).
        if self.reg_weight_vig_non_pos > 0.0:
            alphas = self.vignetting.raw[..., 2:5]  # [C, 3, 3]
            vig_non_pos = F.relu(alphas).mean()
            loss = loss + self.reg_weight_vig_non_pos * vig_non_pos
        # 4. Vig channel: mean(var over RGB channels of vig params).
        # Vig raw: [num_cameras, 3, 5]. Var over channel dim (dim=1).
        if self.reg_weight_vig_channel > 0.0:
            vig_var = self.vignetting.raw.var(dim=1, unbiased=False).mean()
            loss = loss + self.reg_weight_vig_channel * vig_var
        # 5. Color mean: smooth_l1(mean(color @ pinv), beta=0.005) over 8 outputs.
        if self.reg_weight_color_mean > 0.0:
            # color.raw: [num_frames, 8]; pinv_blocks: [4, 2, 2] -> stacked: [8, 8].
            # We need to build an 8x8 matrix that maps the 8 latent dims to 8
            # "interpretable" dims via the pinv blocks. Each (b,r,g,n) pair is
            # processed by its 2x2 block. So the effective 8x8 is block-diagonal
            # with the 4 pinv blocks on the diagonal.
            # Build block-diagonal: [(b)(r)(g)(n)] each is a 2x2 pinv block.
            pinv_diag = torch.block_diag(*[self.color.pinv_blocks[i] for i in range(4)])
            # Apply: color_transformed = color @ pinv_diag.T.
            color_transformed = self.color.raw @ pinv_diag.t()  # [num_frames, 8]
            color_mean_vec = color_transformed.mean(dim=0)  # [8]
            # smooth_l1 over each of 8 dims.
            beta = 0.005
            abs_v = color_mean_vec.abs()
            smooth = torch.where(
                abs_v < beta,
                0.5 * (color_mean_vec / beta) ** 2,
                abs_v - 0.5 * beta,
            )
            loss = loss + self.reg_weight_color_mean * smooth.mean()
        # 6. CRF channel: mean(var over RGB channels of crf params).
        if self.reg_weight_crf_channel > 0.0:
            crf_var = self.crf.raw.var(dim=1, unbiased=False).mean()
            loss = loss + self.reg_weight_crf_channel * crf_var
        return loss

    # ------------------------------------------------------------
    # Optimizer step (manual LR + Adam, gated by train.py)
    # ------------------------------------------------------------

    def zero_grad(self):
        """Zero gradients on all 4 stage parameters."""
        self.optimizer.zero_grad()

    def optimizer_step(self):
        """Apply accumulated gradients via Adam. Caller must zero_grad before
        the next iteration's backward pass. Uses self._current_lr (set by
        scheduler_step), so callers must call scheduler_step() every iter.

        Per-param L2 gradient clipping (max_norm=1.0) bounds per-iter
        param jumps. 2026-08-05 Gate C: without this, raw params can drift
        into the saturation zone (raw > PPISP_MAX_SHAPE_RAW), where
        bounded_positive_forward's softplus saturates and pow overflows.
        LFS doesn't need this because LFS's per-param backward explicitly
        zeros gradient in the saturation zone; PyTorch autograd through
        F.softplus + clamp gives correct saturation but doesn't bound
        drift speed.
        """
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        # Set the optimizer's LR to self._current_lr.
        for pg in self.optimizer.param_groups:
            pg["lr"] = self._current_lr
        self.optimizer.step()
        self._opt_step += 1

    def scheduler_step(self):
        """Advance LR schedule every iter (whether or not engaged).

        This matches the BilateralGrid pattern: schedule tracks iter count,
        not optimizer-step count, so the post-engage window starts from
        a predictable schedule position.
        """
        self._step += 1
        scale = self._lr_lambda(self._step)
        self._current_lr = self.initial_lr * scale
"""High-level wrapper for the fusedssim CUDA op.

The C++ binding returns a 4-tuple of tensors that do NOT carry autograd
metadata (the values are produced by a custom CUDA kernel that the
PyTorch autograd graph does not see). We wrap it in a torch.autograd
Function that re-establishes the gradient path on backward via the
companion `fusedssim_backward` op.

Mirrors the API shape of the upstream gaussian-splatting fused_ssim.
"""
import torch
from fused_ssim_cuda import fusedssim_backward as _fusedssim_backward_op


_C1 = 0.01 ** 2
_C2 = 0.03 ** 2


class _FusedSSIM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, img1, img2, train):
        from fused_ssim_cuda import fusedssim as _op
        target, dm_dmu1, dm_dsigma1_sq, dm_dsigma12 = _op(_C1, _C2, img1, img2, train)
        ctx.save_for_backward(img1, img2, dm_dmu1, dm_dsigma1_sq, dm_dsigma12)
        return target

    @staticmethod
    def backward(ctx, grad_output):
        img1, img2, dm_dmu1, dm_dsigma1_sq, dm_dsigma12 = ctx.saved_tensors
        grad_img1 = _fusedssim_backward_op(
            _C1, _C2, img1, img2, grad_output, dm_dmu1, dm_dsigma1_sq, dm_dsigma12,
        )
        return grad_img1, None, None


def _fusedssim(img1, img2, train=True):
    return _FusedSSIM.apply(img1, img2, train)


def fusedssim(img1, img2, pad=3, window_size=11, size_average=True, set_signed=False):
    """SSIM between two images.

    Args:
        img1, img2: (B, C, H, W) float tensors in [0, 1].
        size_average: if True, returns scalar SSIM; else per-batch map.
        set_signed: unused here (kept for API parity); see upstream.

    Returns:
        Tensor: scalar if size_average else (1, B) per-batch SSIM.
    """
    target = _fusedssim(img1, img2, train=True)
    if size_average:
        return target.mean()
    return target.mean(dim=(2, 3)).squeeze(0)


__all__ = ["fusedssim", "fusedssim_backward"]

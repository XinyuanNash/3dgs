#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree, optimizer_type="default"):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        # P1-MultiStrategy (OPTIMIZATIONS.md §13.6 + §13.7): per-Gaussian
        # visibility counter incremented by MCMC strategy each post_backward
        # call (track how often each Gaussian has been seen during training).
        # Initialized as empty (zero-size) here; training_setup gives it shape
        # (N, 1) once the model has Gaussians.
        self.opacity_visible_count = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        # P1-7+ (OPTIMIZATIONS.md §13.6): persistent per-Gaussian pixel-error
        # buffer handle. handle=0 means "no buffer" (legacy bit-identical
        # path); train.py calls ensure_error_state() to obtain a real handle
        # when --p1_7_plus is set. The buffer is owned by the C++ side; the
        # model just holds the handle.
        self.error_state_handle = 0
        self._error_state_initialized = False
        self.setup_functions()

    def ensure_error_state(self):
        """P1-7+: lazily create the rasterizer's persistent error buffer.

        Idempotent. Returns the opaque int handle (=0 if creation was skipped
        because the model is empty or a previous call already created one).
        """
        if not getattr(self, "_error_state_initialized", False):
            try:
                from diff_gaussian_rasterization import _C
                self.error_state_handle = int(_C.make_rasterizer_state())
                self._error_state_initialized = True
            except Exception:
                # Module not loaded or rasterizer too old — leave handle=0.
                self.error_state_handle = 0
        return self.error_state_handle

    def free_error_state(self):
        """P1-7+: free the rasterizer error buffer. Idempotent."""
        if getattr(self, "_error_state_initialized", False):
            try:
                from diff_gaussian_rasterization import _C
                _C.free_rasterizer_state(int(self.error_state_handle))
            except Exception:
                pass
            self.error_state_handle = 0
            self._error_state_initialized = False

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            # P1-MultiStrategy (OPTIMIZATIONS.md §13.7): per-Gaussian
            # visibility counter (MCMC bookkeeping). Appended at the END
            # so existing P0-WarpBallot checkpoints load via the legacy
            # 12-tuple path in restore() below.
            self.opacity_visible_count,
        )

    def restore(self, model_args, training_args):
        (self.active_sh_degree,
         self._xyz,
         self._features_dc,
         self._features_rest,
         self._scaling,
         self._rotation,
         self._opacity,
         self.max_radii2D,
         xyz_gradient_accum,
         denom,
         opt_dict,
         self.spatial_lr_scale,
         *rest) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
        # Backwards-compat: pre-P1-3 checkpoints have a 12-tuple; the 13th
        # element (opacity_visible_count) defaults to a fresh zero buffer
        # of the current model size.
        if rest and rest[0] is not None:
            self.opacity_visible_count = rest[0]
        else:
            self.opacity_visible_count = torch.zeros(
                (self.get_xyz.shape[0], 1), device="cuda"
            )

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_exposure(self):
        return self._exposure

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, cam_infos : int, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        # P1-MultiStrategy (OPTIMIZATIONS.md §13.7): per-Gaussian visibility
        # counter for MCMC. (N, 1) keeps shape consistent with accum/denom
        # so densification_postfix + prune_points can reset/slice without
        # changing the buffer's dtype/layout.
        self.opacity_visible_count = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        # P1-MultiStrategy (OPTIMIZATIONS.md §13.7): tmp_radii is normally
        # only allocated by densify_and_prune, but MCMC clone/relocate also
        # touches it (LFS mcmc.cpp does too). Initialise here so the
        # attribute exists even before the first densify_and_prune call —
        # MCMC may need to clone before any densification has happened.
        self.tmp_radii = torch.zeros((self.get_xyz.shape[0],), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                # A special version of the rasterizer is required to enable sparse adam
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.exposure_optimizer = torch.optim.Adam([self._exposure])

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
                                                        lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                        lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                        max_steps=training_args.iterations)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        # Defensive NaN reset for downstream tools (e.g. gsbox ply2sog).
        # A small number of Gaussians (~1/1.2M) may end up with NaN
        # xyz/scale/rotation from rare GPU numerical cases during
        # densification/Adam — opacity & SH stay finite. We only touch
        # the geometric params, in-place via .data (bypasses autograd
        # and Adam state). Training is over by the time save_ply runs,
        # so Adam state being out of sync with the reset tensors is
        # irrelevant — nothing will read those state tensors again.
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        bad = (np.isnan(xyz).any(axis=1)
               | np.isnan(scale).any(axis=1)
               | np.isnan(rotation).any(axis=1))
        if bad.any():
            n_bad = int(bad.sum())
            print(f"[save_ply] {n_bad} Gaussian(s) have NaN xyz/scale/rot; "
                  f"resetting to identity for PLY output")
            self._xyz.data[bad] = 0.0
            # _scaling is pre-exp; 0 -> exp(0) = 1 (sensible default)
            self._scaling.data[bad] = 0.0
            # _rotation is pre-normalize; [1,0,0,0] -> identity quat
            identity_quat = torch.tensor(
                [1., 0., 0., 0.], device=self._rotation.device,
                dtype=self._rotation.dtype)
            self._rotation.data[bad] = identity_quat
            # Re-read after the in-place reset so the np.empty buffer
            # below sees the cleaned values, not the originals.
            xyz = self._xyz.detach().cpu().numpy()
            scale = self._scaling.detach().cpu().numpy()
            rotation = self._rotation.detach().cpu().numpy()

        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, use_train_test_exp = False):
        plydata = PlyData.read(path)
        if use_train_test_exp:
            exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
                print(f"Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    # ----------------------------------------------------------------------
    # P1-MultiStrategy thin wrappers (OPTIMIZATIONS.md §13.7).
    # These loop over the model's 6 param groups and delegate to the
    # low-level primitives in ``scene.strategy.adam_state_ops``. MCMC
    # (P1-3) and IGS+ (P1-4/5) call these to manage Adam moments when
    # Gaussians are cloned / relocated.
    # ----------------------------------------------------------------------

    #: Name -> index in ``self.optimizer.param_groups``. Frozen at import time
    #: so wrapper methods don't have to do a string match on every call.
    _ADAM_PARAM_NAMES = ("xyz", "f_dc", "f_rest", "opacity", "scaling", "rotation")

    #: Map optimizer-group name → GaussianModel attribute. ``f_dc`` and
    #: ``f_rest`` use ``_features_dc`` / ``_features_rest`` (NOT the
    #: ``_f_dc`` / ``_f_rest`` shorthand).
    _ADAM_PARAM_TO_ATTR = {
        "xyz":      "_xyz",
        "f_dc":     "_features_dc",
        "f_rest":   "_features_rest",
        "opacity":  "_opacity",
        "scaling":  "_scaling",
        "rotation": "_rotation",
    }

    def reset_state(self, indices):
        """Zero Adam state at the given indices across all 6 param groups.

        See :func:`scene.strategy.adam_state_ops.reset_state_at_indices`
        for the full contract.
        """
        from scene.strategy.adam_state_ops import reset_state_at_indices
        idx = torch.as_tensor(indices, dtype=torch.int64).to("cuda")
        for name in self._ADAM_PARAM_NAMES:
            reset_state_at_indices(self.optimizer, name, idx)

    def relocate_state(self, src_indices, dst_indices):
        """Copy Adam state src -> dst, then zero src, across all 6 param groups.

        See :func:`scene.strategy.adam_state_ops.relocate_state_at_indices`
        for the full contract.
        """
        from scene.strategy.adam_state_ops import relocate_state_at_indices
        src = torch.as_tensor(src_indices, dtype=torch.int64).to("cuda")
        dst = torch.as_tensor(dst_indices, dtype=torch.int64).to("cuda")
        for name in self._ADAM_PARAM_NAMES:
            relocate_state_at_indices(self.optimizer, name, src, dst)

    def add_state_zeros(self, n_new):
        """Grow all 6 param tensors by ``n_new`` rows of zeros with zero Adam
        state. Used by callers that want to append fresh-Gaussian slots and
        then fill in param values via direct indexing.

        MCMC (P1-3) does NOT use this — its clone path goes through the
        existing ``densification_postfix`` which copies parent values
        AND zeros new state in one call. This helper exists for callers
        that need the "grow with zeros" pattern explicitly.

        See ``scene.strategy.adam_state_ops`` for state-mutation details.
        """
        n_new = int(n_new)
        if n_new <= 0:
            return
        device = self._xyz.device
        d = {
            "xyz":      torch.zeros((n_new, 3), device=device),
            "f_dc":     torch.zeros((n_new, 1, 3), device=device),
            "f_rest":   torch.zeros((n_new, (self.max_sh_degree + 1) ** 2 - 1, 3), device=device),
            "opacity":  torch.zeros((n_new, 1), device=device),
            "scaling":  torch.zeros((n_new, 3), device=device),
            "rotation": torch.zeros((n_new, 4), device=device),
        }
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz      = optimizable_tensors["xyz"]
        self._features_dc   = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity  = optimizable_tensors["opacity"]
        self._scaling  = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        # Also grow the MCMC visibility counter and tmp_radii so the new
        # slots are valid for downstream surgery. tmp_radii defaults to 0
        # (will be set by densify_and_prune on the next refine cycle).
        self.opacity_visible_count = torch.zeros(
            (self.get_xyz.shape[0], 1), device=device
        )
        self.tmp_radii = torch.zeros((self.get_xyz.shape[0],), device=device)

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.tmp_radii = self.tmp_radii[valid_points_mask]
        # P1-MultiStrategy (OPTIMIZATIONS.md §13.7): drop the visibility
        # count for pruned Gaussians alongside xyz_gradient_accum/denom.
        # Use the same valid_points_mask so the buffer stays aligned with
        # the param tensors.
        self.opacity_visible_count = self.opacity_visible_count[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        # P1-MultiStrategy (OPTIMIZATIONS.md §13.7): cloned/split Gaussians
        # start with zero visibility. Their count then ticks up under the
        # MCMC post_backward loop like everyone else's.
        self.opacity_visible_count = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        # P1-MultiStrategy precondition guard (see STATUS_P0-LAS.md note +
        # OPTIMIZATIONS.md §13.7). densify_and_split consumes self.tmp_radii
        # to seed the new tmp_radii for the parent + child Gaussians. Calling
        # it without first populating tmp_radii (via densify_and_prune) leaves
        # the new Gaussians with stale radii from a prior iteration's frame.
        # Make that misuse loud rather than silently producing wrong state.
        assert self.tmp_radii is not None, (
            "densify_and_split requires self.tmp_radii to be set first; "
            "call densify_and_prune(max_grad, min_opacity, extent, max_screen_size, radii) "
            "or assign self.tmp_radii = radii explicitly before densifying."
        )
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        # === [P0-LAS] Long-Axis Split ===
        # Reference: LichtFeld-Studio/src/training/kernels/densification_kernels.cu:674-806
        # OPTIMIZATIONS.md §1.2
        # Replaces isotropic Gaussian noise sampling (old 3DGS) with projection
        # along each Gaussian's longest local axis. Preserves anisotropy and
        # avoids the "spherical bump" artifact of the original split.
        sel_scaling = self.get_scaling[selected_pts_mask]                # [M, 3]
        sel_xyz     = self._xyz[selected_pts_mask]                       # [M, 3]
        sel_rot     = self._rotation[selected_pts_mask]                  # [M, 4]
        sel_opacity = self._opacity[selected_pts_mask]                   # [M, 1]

        # 1. Longest local axis per selected Gaussian (0/1/2)
        longest_idx = sel_scaling.argmax(dim=-1)                         # [M]

        # 2. offset magnitude = exp(scale[longest]) * 0.5
        sel_longest_scale = sel_scaling.gather(1, longest_idx.unsqueeze(-1)).squeeze(-1)  # [M]
        offset_mag = sel_longest_scale * 0.5                             # [M]

        # 3. New scaling in log space:
        #    longest axis:  scale *= 0.5  (log += log(0.5))
        #    other axes:   scale *= 0.85 (log += log(0.85))
        # Matches LFS densification_kernels.cu:723-726.
        # NOTE: operate on log-scaled parameters (self._scaling) so the offsets
        # in log space correctly correspond to multiplicative scale changes.
        sel_log_scaling = self._scaling[selected_pts_mask]               # [M, 3]
        log_half = np.log(0.5)
        log_085  = np.log(0.85)
        new_log_scaling = sel_log_scaling + log_085                      # default for "other" axes
        new_log_scaling.scatter_(
            1,
            longest_idx.unsqueeze(-1),
            sel_log_scaling.gather(1, longest_idx.unsqueeze(-1)) + log_half,
        )

        # 4. New opacity: inv_sigmoid(sigmoid(opacity) * 0.6)
        # Match LFS:  raw_sig = sig * 0.6;  new_opacity = inverse_sigmoid(raw_sig)
        # Clamp to keep the logit finite when the source opacity is very small.
        raw_sig = torch.sigmoid(sel_opacity) * 0.6
        new_opacity_logit = self.inverse_opacity_activation(
            torch.clamp(raw_sig, min=1e-6, max=1.0 - 1e-6)
        )

        # 5. World-space offset direction = R[i] @ e_{longest_idx[i]}
        #    (column `longest_idx` of the per-Gaussian rotation matrix)
        R = build_rotation(sel_rot)                                      # [M, 3, 3]
        offset_dir = R.gather(
            2, longest_idx.view(-1, 1, 1).expand(-1, 3, 1)
        ).squeeze(-1)                                                    # [M, 3]
        offset_world = offset_dir * offset_mag.unsqueeze(-1)             # [M, 3]

        # 6. Parent = pos + offset  (overwrites src position, in-place)
        #    Child  = pos - offset  (new Gaussian)
        parent_xyz = sel_xyz + offset_world                              # [M, 3]
        child_xyz  = sel_xyz - offset_world                              # [M, 3]
        new_xyz = torch.cat([parent_xyz, child_xyz], dim=0)              # [2M, 3]

        # Repeat other attributes to [2M, ...]
        new_log_scaling_rep = new_log_scaling.repeat(N, 1)               # [2M, 3]
        new_opacity_rep     = new_opacity_logit.repeat(N, 1)             # [2M, 1]
        new_rotation        = sel_rot.repeat(N, 1)                       # [2M, 4]
        new_features_dc     = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest   = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_tmp_radii       = self.tmp_radii[selected_pts_mask].repeat(N)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest,
                                   new_opacity_rep, new_log_scaling_rep,
                                   new_rotation, new_tmp_radii)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # P1-MultiStrategy precondition guard (mirror densify_and_split above;
        # see STATUS_P0-LAS.md + OPTIMIZATIONS.md §13.7).
        assert self.tmp_radii is not None, (
            "densify_and_clone requires self.tmp_radii to be set first; "
            "call densify_and_prune(max_grad, min_opacity, extent, max_screen_size, radii) "
            "or assign self.tmp_radii = radii explicitly before densifying."
        )
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_tmp_radii = self.tmp_radii[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.tmp_radii = radii
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        tmp_radii = self.tmp_radii
        self.tmp_radii = None

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    # ----------------------------------------------------------------------
    # P3-Morton (doc/specs/P3-Morton-spec.md §2.3-2.4): Morton-order
    # permutation for GPU cache locality. Reorders ALL Gaussian tensors
    # (params + densification buffers + Adam moments) by Morton code of
    # ``_xyz``. Pure data layout — does NOT affect loss / gradient.
    #
    # Default OFF (no caller invokes this until train.py honors
    # ``--use_morton_order``). When ON, called periodically from train.py
    # during the densify window only (per spec §2.4 / §6).
    #
    # Trap (per [[decoupling-train-py-refactor-committed]] gotcha pattern):
    #   Adam state (per-param exp_avg / exp_avg_sq) MUST be permuted in
    #   lockstep with the param tensor; otherwise optimizer.step applies
    #   momentum from a different Gaussian to the current slot → NaN /
    #   divergence. Densification auxiliary buffers (max_radii2D /
    #   xyz_gradient_accum / denom / tmp_radii / opacity_visible_count)
    #   also need permuting or downstream surgery will index into the
    #   wrong Gaussian. SORTED_BUFFERS + ADAM_STATE_KEYS below pin the
    #   full list — keep in sync with anything that adds a new buffer.
    # ----------------------------------------------------------------------

    #: Buffers permuted along dim 0 by ``apply_permutation``. Each name
    #: matches an attribute on ``self``. None entries are tolerated (a
    #: buffer not yet allocated, e.g. tmp_radii before the first
    #: densify_and_prune call) and skipped silently.
    SORTED_BUFFERS = (
        "_xyz", "_scaling", "_rotation", "_opacity",
        "_features_dc", "_features_rest",
        "max_radii2D", "xyz_gradient_accum", "denom", "tmp_radii",
        "opacity_visible_count",
    )

    #: Names matching ``self.optimizer.param_groups[*]['name']`` whose
    #: Adam state must be permuted. The state is keyed by
    #: ``id(param)`` — we look up each param, then mutate the stored
    #: ``exp_avg`` / ``exp_avg_sq`` in place via ``tensor[perm]``.
    ADAM_STATE_KEYS = ("xyz", "f_dc", "f_rest", "opacity", "scaling", "rotation")

    def apply_permutation(self, perm: torch.Tensor) -> None:
        """Apply ``perm`` along dim 0 to every Gaussian tensor + Adam state.

        ``perm`` is an int64 tensor of shape ``[N]`` that, when used as
        an indexer, reorders the first dim (e.g. ``x[perm]``). Buffers
        whose leading dim size doesn't match ``perm.shape[0]`` are
        skipped (they're stale buffers from a prior iter / not yet
        allocated). Adam state groups with no state dict entry yet
        (init-only params) are skipped — same defensive posture as
        ``_prune_optimizer``.

        Implementation note: we MUST re-register the Adam state under
        the NEW Parameter's id, because ``self._xyz = self._xyz[perm]``
        replaces the Parameter with a fresh tensor (different Python
        id). Without the re-register, ``optimizer.state[id(new_xyz)]``
        returns nothing → next ``optimizer.step`` would silently
        reinitialize state from scratch → learning-rate trajectory
        corruption (visible as PSNR collapse at iter ~1000 even with
        correctly-permuted data). This is the same pattern as
        ``_prune_optimizer`` and ``replace_tensor_to_optimizer`` above.
        """
        n = int(perm.shape[0])
        # IMPORTANT: ``buf[perm]`` returns a non-leaf Tensor (with
        # ``grad_fn=IndexBackward``). Adam reads ``param.grad`` to
        # step — non-leaf tensors NEVER get ``.grad`` populated by
        # ``loss.backward()`` (gradient flows to the source leaf
        # instead, which is the OLD tensor no longer attached to
        # ``self``). The fix is to wrap in ``nn.Parameter(...)``,
        # which discards ``grad_fn`` and yields a fresh leaf with
        # ``requires_grad=True``. Same pattern as ``_prune_optimizer``
        # and ``replace_tensor_to_optimizer`` above.
        for name in self.SORTED_BUFFERS:
            buf = getattr(self, name, None)
            if buf is None:
                continue
            if not hasattr(buf, "shape") or buf.shape[0] != n:
                continue
            # nn.Parameter for the 6 Adam-tracked params; plain Tensor
            # for densification scratch buffers (max_radii2D /
            # xyz_gradient_accum / denom / tmp_radii / opacity_visible_count)
            # since they don't need grad.
            if name in self._ADAM_PARAM_TO_ATTR.values():
                setattr(self, name,
                        nn.Parameter(buf[perm].requires_grad_(True)))
            else:
                setattr(self, name, buf[perm])

        # Permute Adam state in lockstep with the params. For each
        # param-group, we look up the OLD param tensor (the one whose
        # id is currently in optimizer.state), permute its exp_avg /
        # exp_avg_sq, then DELETE the old state entry and CREATE a
        # new one under the NEW param tensor's id (which is what
        # ``setattr(self, name, buf[perm])`` just installed).
        if self.optimizer is not None:
            for group in self.optimizer.param_groups:
                gname = group.get("name")
                if gname not in self.ADAM_STATE_KEYS:
                    continue
                old_param = group["params"][0]
                state = self.optimizer.state.get(old_param, None)
                if state is None:
                    continue
                # Permute Adam moments in place if their leading dim
                # matches perm (it always should, since they shadow
                # the param shape). exp_avg / exp_avg_sq may be missing
                # for init-only params; guard explicitly.
                for key in ("exp_avg", "exp_avg_sq"):
                    t = state.get(key, None)
                    if t is None:
                        continue
                    if hasattr(t, "shape") and t.shape[0] == n:
                        state[key] = t[perm]
                # Re-register state under the NEW param's id. The
                # new param is whatever attr `_<gname>` now points to.
                attr = self._ADAM_PARAM_TO_ATTR.get(gname)
                if attr is None:
                    continue
                new_param = getattr(self, attr, None)
                if new_param is None or new_param is old_param:
                    # Param identity unchanged (shouldn't happen with
                    # param reassignment, but defend against it).
                    continue
                # Move the state dict entry to the new id.
                if old_param in self.optimizer.state:
                    del self.optimizer.state[old_param]
                self.optimizer.state[new_param] = state
                # Update the param-group's param reference too (so
                # optimizer.step() actually steps the new tensor).
                group["params"][0] = new_param

    def reorder_morton(self, strategy=None) -> None:
        """Reorder all Gaussian state by Morton order of ``_xyz``.

        Called by train.py every ``--morton_reorder_interval`` iters
        inside the densify window when ``--use_morton_order`` is on.
        Pure permute — does not modify any param value, only shuffles
        which index each Gaussian occupies.

        Args:
            strategy: Optional :class:`Strategy` whose per-Gaussian
                state (IGS+ _free_mask/_error_score_max/_edge_score_cache,
                MCMC _error_score_max) should be permuted by the same
                permutation. Pass ``None`` (default) to skip the
                strategy permute — used by callers that have already
                permuted the strategy or that don't have one.

        No-op when the model has zero Gaussians (morton_encode_3d on an
        empty [0, 3] tensor would assert on the bbox-normalization min
        step). Also a no-op when N < ``--morton_min_gaussians`` to
        skip the overhead on small datasets (campus 800-iter
        ~50K Gaussians → skip).
        """
        n = int(self._xyz.shape[0])
        if n == 0:
            return
        # Lazy import to keep gaussian_model import-cost minimal for
        # unit tests that don't care about Morton.
        from utils.morton import morton_encode_3d, morton_sort_indices
        codes = morton_encode_3d(self._xyz.detach())
        perm = morton_sort_indices(codes)
        self.apply_permutation(perm)
        # Thread the same perm into the strategy so its slot-indexed
        # state stays aligned with the Gaussians. DefaultStrategy has
        # no per-Gaussian state and the no-op apply_permutation runs in
        # ~1µs; IGS+ / MCMC permute their score buffers.
        if strategy is not None:
            strategy.apply_permutation(perm)

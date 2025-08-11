from __future__ import annotations
import time
import torch
import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), './'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from forward_occupancy.SO import make_spheres

from planning.splanning.splat_tools import SplatLoader
from planning.splanning.gaussian_integrator import integrate_gaussians_spheres

from enum import Enum
import time

class ConstraintAggregationMode(Enum):
    NONE=0
    TIME_INTERVALS=1
    ALL=2

class SplanningConstraints:
    _integrator = None
    def __init__(self, dimension = 3, dtype = torch.float, device=None, 
                 max_spheres=5000, n_params=7, 
                 self_collision_top_k=5,
                 splat_loader: SplatLoader = None,
                 constraint_aggregation_mode = ConstraintAggregationMode.TIME_INTERVALS,
                 constraint_rhs = 0.05,
                 constraint_alpha=0.05):
        self.dimension = dimension
        self.n_params = n_params
        self.dtype = dtype
        self.np_dtype = torch.empty(0,dtype=dtype).numpy().dtype
        if device is None:
            device = torch.empty(0,dtype=dtype).device
        self.device = device
        self.top_k = self_collision_top_k # hyperparameter, 5-10 seems to be a good number
        self.alpha = constraint_alpha

        self.constraint_aggregation_mode = constraint_aggregation_mode
        self.constraint_rhs = constraint_rhs

        self.splats: SplatLoader = splat_loader

        if self.splats is not None and self.splats.sigma_magnification is None:
            self.splats.precompute_errors_for_sigma_mag(6)

        self.__allocate_base_params(max_spheres)

        # self.integrator = integrate_gaussians_spheres
        if SplanningConstraints._integrator is None:
            try:
                SplanningConstraints._integrator = torch.compile(integrate_gaussians_spheres, dynamic=True, options={'shape_padding': True, 'epilogue_fusion': True, 'max_autotune': True})
            except:
                print("Failed to compile the integration. Falling back to the super slow mode")
                SplanningConstraints._integrator = integrate_gaussians_spheres

        self._has_warned_splat = False

    def __allocate_base_params(self, max_spheres):
        self.max_spheres = max_spheres
        self.__centers = torch.empty((self.max_spheres, self.dimension), dtype=self.dtype, device=self.device)
        self.__radii = torch.empty((self.max_spheres), dtype=self.dtype, device=self.device)
        self.__center_jac = torch.empty((self.max_spheres, self.dimension, self.n_params), dtype=self.dtype, device=self.device)
        self.__radii_jac = torch.empty((self.max_spheres, self.n_params), dtype=self.dtype, device=self.device)

    def set_params(self, centers_bpz, radii, p_idx, obs_link_mask, g_ka, n_spheres_per_link, n_time, obs_tuple, self_collision, gaussian_mask):
        self.centers_bpz = centers_bpz
        self.p_idx = p_idx
        self.obs_link_mask = obs_link_mask
        self.g_ka = g_ka
        n_joints = centers_bpz.batch_shape[0]
        self.n_joints = n_joints
        self.n_spheres = n_spheres_per_link
        self.n_time = n_time

        ### Self collision checks
        if self_collision is not None:
            self.self_collision_link_link = self_collision[0]
            self.self_collision_link_joint = self_collision[1]
            self.self_collision_joint_joint = self_collision[2]
            self.n_link_link = self.self_collision_link_link.count_nonzero(dim=1).cpu()
            self.n_link_joint = self.self_collision_link_joint.count_nonzero(dim=1).cpu()
            self.n_joint_joint = self.self_collision_joint_joint.count_nonzero(dim=1).cpu()
            num_link_link = self.n_link_link.sum()
            num_link_joint = self.n_link_joint.sum()
            num_joint_joint = self.n_joint_joint.sum()
            # self.n_self_collision = num_link_link*n_time*n_spheres_per_link*n_spheres_per_link \
            #                         + num_link_joint*n_time*n_spheres_per_link \
            #                         + num_joint_joint*n_time
            self.n_self_collision_full = num_link_link*n_spheres_per_link*n_spheres_per_link \
                                    + num_link_joint*n_spheres_per_link \
                                    + num_joint_joint
            self.n_self_collision = self.top_k*n_time
        else:
            self.n_self_collision = 0
        ###

        n_pairs = p_idx.shape[-1]
        if obs_link_mask is not None:
            if self.n_self_collision == 0:
                self.p_idx = p_idx[:,obs_link_mask]
            n_obs_link_pairs = torch.count_nonzero(obs_link_mask).item()
        else:
            n_obs_link_pairs = n_pairs
            
        self.total_spheres = n_joints*n_time + n_spheres_per_link*n_obs_link_pairs*n_time

        
        if self.total_spheres > self.max_spheres:
            # reallocate new tensors
            self.__allocate_base_params(self.total_spheres)

        if self.constraint_aggregation_mode == ConstraintAggregationMode.NONE:
            self.M = self.total_spheres + self.n_self_collision
        elif self.constraint_aggregation_mode == ConstraintAggregationMode.TIME_INTERVALS:
            self.M = self.n_time + self.n_self_collision
        elif self.constraint_aggregation_mode == ConstraintAggregationMode.ALL:
            self.M = 1 + self.n_self_collision
        else:
            raise RuntimeError(f"Unknown constraint aggregation mode {self.constraint_aggregation_mode}")
            
        self.n_obs_in_FO = 1
        ###

        ## Utilize underlying storage and update initial radii values
        self.centers = self.__centers[:self.total_spheres]
        self.radii = self.__radii[:self.total_spheres]
        self.center_jac = self.__center_jac[:self.total_spheres]
        self.radii_jac = self.__radii_jac[:self.total_spheres]

        self.radii.view(-1, self.n_time)[:n_joints] = radii
        self.radii_jac[:self.n_time*n_joints] = 0

        ## Obstacle data
        self.obs_tuple = obs_tuple


        if self.splats is None:
            if not self._has_warned_splat:
                print("No splat. Skipping all remaining setup. This will be a problem unless you're only doing visualization")
            self._has_warned_splat = True
            return

        self.gaussian_mask = gaussian_mask
        inv_mask = ~self.gaussian_mask
        self.gauss_errors = self.splats.splat_errors[inv_mask].sum()


    def __self_collision(self, spheres, joint_centers, joint_radii, joint_jacs, Cons_out, Jac_out):
        '''This function is used to compute the distance and gradient of the distance to each other sphere that is valid for self-collision.

        Args:
            spheres: A tuple of tensors representing the centers and radii of each sphere and their respective jacobians.
            joint_centers: A tensor of shape (n_joints, n_time, dim) representing the centers of each joint.
            joint_radii: A tensor of shape (n_joints, n_time) representing the radii of each joint.
            joint_jacs: A tensor of shape (n_joints, n_time, dim, n_params) representing the jacobians of each joint.
            Cons_out: An output tensor of shape (n_self_collision) representing the constraint values. This is modified in place.
            Jac_out: An output tensor of shape (n_self_collision, n_params) representing the jacobian of the constraint values. This is modified in place.
        '''
        # First get each link to link distance
        # spheres[0] are (link_idx, time_idx, sphere_idx, dim)
        # spheres[1] are (link_idx, time_idx, sphere_idx)
        # link_spheres_c are (time_idx, sphere_idx, dim)
        # link_spheres_r are (time_idx, sphere_idx)
        Cons_out_ = torch.empty((self.n_time, self.n_self_collision_full), dtype=self.dtype, device=self.device)
        Jac_out_ = torch.empty((self.n_time, self.n_self_collision_full, self.n_params), dtype=self.dtype, device=self.device)
        sidx = 0
        for link_idx, (link_spheres_c, link_spheres_r, jac_c, jac_r) in enumerate(zip(*spheres)):
            if self.n_link_link[link_idx] == 0:
                continue
            comp_spheres_c = spheres[0][self.self_collision_link_link[link_idx]] # (n_comp_links, time_idx, sphere_idx, dim)
            comp_spheres_r = spheres[1][self.self_collision_link_link[link_idx]] # (n_comp_links, time_idx, sphere_idx)
            delta = link_spheres_c[None,...,None,:] - comp_spheres_c[...,None,:,:]
            dists = torch.linalg.vector_norm(delta, dim=-1)
            surf_dists = dists - link_spheres_r.unsqueeze(-1) - comp_spheres_r.unsqueeze(-2) # (n_comp_links, time_idx, sphere_idx(self), sphere_idx)
            # compute jacobian
            # delta is (n_comp_links, time_idx, sphere_idx(self), sphere_idx, dim)
            # jac_c is (n_comp_links, time_idx, sphere_idx(self), dim, n_params)
            # jac_r is (n_comp_links, time_idx, sphere_idx(self), n_params)
            comp_spheres_jac_c = spheres[2][self.self_collision_link_link[link_idx]]
            comp_spheres_jac_r = spheres[3][self.self_collision_link_link[link_idx]]
            # (n_comp_links, time_idx, sphere_idx(self), sphere_idx, dim, n_params)
            jac_dists_inner = (delta/dists.unsqueeze(-1)).unsqueeze(-1) * (jac_c[None,...,None,:,:] - comp_spheres_jac_c[...,None,:,:,:])
            # (n_comp_links, time_idx, sphere_idx(self), sphere_idx, n_params)
            jac_dists = torch.sum(jac_dists_inner, dim=-2)
            jac_surf_dists = jac_dists - jac_r.unsqueeze(-2) - comp_spheres_jac_r.unsqueeze(-3)
            # Save
            eidx = sidx + surf_dists.numel()//self.n_time
            Cons_out_[:,sidx:eidx].copy_(-surf_dists.transpose(0,1).reshape(self.n_time, -1))
            Jac_out_[:,sidx:eidx].copy_(-jac_surf_dists.transpose(0,1).reshape(self.n_time, -1, self.n_params))
            sidx = eidx

        # Then get each link to joint distance
        for link_idx, (link_spheres_c, link_spheres_r, jac_c, jac_r) in enumerate(zip(*spheres)):
            if self.n_link_joint[link_idx] == 0:
                continue
            comp_spheres_c = joint_centers[self.self_collision_link_joint[link_idx]].unsqueeze(-2) # (n_comp_links, time_idx, 1, dim)
            comp_spheres_r = joint_radii[self.self_collision_link_joint[link_idx]].unsqueeze(-1) # (n_comp_links, time_idx, 1)
            delta = link_spheres_c.unsqueeze(0) - comp_spheres_c
            dists = torch.linalg.vector_norm(delta, dim=-1)
            surf_dists = dists - comp_spheres_r - link_spheres_r # (n_comp_links, time_idx, sphere_idx)
            # compute jacobian
            # delta is (n_comp_links, time_idx, sphere_idx, dim)
            # jac_c is (n_comp_links, time_idx, sphere_idx, dim, n_params)
            # jac_r is (n_comp_links, time_idx, sphere_idx, n_params)
            comp_spheres_jac_c = joint_jacs[self.self_collision_link_joint[link_idx]]
            # (n_comp_links, time_idx, sphere_idx, dim, n_params)
            jac_dists_inner = (delta/dists.unsqueeze(-1)).unsqueeze(-1) * (jac_c.unsqueeze(0) - comp_spheres_jac_c.unsqueeze(-3))
            # (n_comp_links, time_idx, sphere_idx, n_params)
            jac_dists = torch.sum(jac_dists_inner, dim=-2)
            jac_surf_dists = jac_dists - jac_r.unsqueeze(0)
            # Save
            eidx = sidx + surf_dists.numel()//self.n_time
            Cons_out_[:,sidx:eidx].copy_(-surf_dists.transpose(0,1).reshape(self.n_time, -1))
            Jac_out_[:,sidx:eidx].copy_(-jac_surf_dists.transpose(0,1).reshape(self.n_time, -1, self.n_params))
            sidx = eidx

        # Then get each joint to joint distance
        # joint_centers_all are (joint_idx, time_idx, dim)
        # joint_radii_all are (joint_idx, time_idx)
        for joint_idx, (joint_spheres_c, joint_spheres_r, joint_spheres_jac) in enumerate(zip(joint_centers, joint_radii, joint_jacs)):
            if self.n_joint_joint[joint_idx] == 0:
                continue
            comp_spheres_c = joint_centers[self.self_collision_joint_joint[joint_idx]] # (n_comp_joints, time_idx, dim)
            comp_spheres_r = joint_radii[self.self_collision_joint_joint[joint_idx]] # (n_comp_joints, time_idx)
            delta = joint_spheres_c.unsqueeze(0) - comp_spheres_c
            dists = torch.linalg.vector_norm(delta, dim=-1)
            surf_dists = dists - comp_spheres_r - joint_spheres_r # (n_comp_joints, time_idx)
            # compute jacobian
            # delta is (n_comp_links, time_idx, dim)
            # jac_c is (n_comp_links, time_idx, dim, n_params)
            # jac_r is (n_comp_links, time_idx, n_params)
            comp_spheres_jac_c = joint_jacs[self.self_collision_joint_joint[joint_idx]]
            # (n_comp_links, time_idx, dim, n_params)
            jac_dists_inner = (delta/dists.unsqueeze(-1)).unsqueeze(-1) * (joint_spheres_jac.unsqueeze(0) - comp_spheres_jac_c)
            # (n_comp_links, time_idx, n_params)
            jac_surf_dists = torch.sum(jac_dists_inner, dim=-2)
            # Save
            eidx = sidx + surf_dists.numel()//self.n_time
            Cons_out_[:,sidx:eidx].copy_(-surf_dists.transpose(0,1).reshape(self.n_time, -1))
            Jac_out_[:,sidx:eidx].copy_(-jac_surf_dists.transpose(0,1).reshape(self.n_time, -1, self.n_params))
            sidx = eidx
        
        surf_dists, surf_dists_idx = torch.topk(Cons_out_, self.top_k, dim=1, sorted=False)
        jac_surf_dists = Jac_out_.gather(1,surf_dists_idx[...,None].expand(-1,-1,self.n_params))
        Cons_out.copy_(surf_dists.reshape(-1))
        Jac_out.copy_(jac_surf_dists.reshape(-1,self.n_params))

    def __call__(self, x, Cons_out=None, Jac_out=None):
        x = torch.as_tensor(x, dtype=self.dtype, device=self.device)
        if Cons_out is None:
            Cons_out = np.empty(self.M, dtype=self.np_dtype)
        if Jac_out is None:
            Jac_out = np.empty((self.M, self.n_params), dtype=self.np_dtype)
        Cons_out = torch.from_numpy(Cons_out)
        Jac_out = torch.from_numpy(Jac_out)

        if self.splats is None:
            Cons_out.zero_()
            Jac_out.zero_()
            return Cons_out, Jac_out

        with torch.no_grad():
            # Batch form of batch construction
            centers = self.centers.view(-1, self.n_time, self.dimension)
            center_jac = self.center_jac.view(-1, self.n_time, self.dimension, self.n_params)
            radii = self.radii.view(-1, self.n_time)
            centers[:self.n_joints] = self.centers_bpz.center_slice_all_dep(x)
            center_jac[:self.n_joints] = self.centers_bpz.grad_center_slice_all_dep(x)
            joint_centers = centers[self.p_idx]
            joint_radii = radii[self.p_idx]
            joint_jacs = center_jac[self.p_idx]
            spheres = make_spheres(joint_centers[0], joint_centers[1], joint_radii[0], joint_radii[1], joint_jacs[0], joint_jacs[1], self.n_spheres)
            sidx = self.n_joints * self.n_time
            if self.n_self_collision > 0 and self.obs_link_mask is not None:
                self.centers[sidx:] = spheres[0][self.obs_link_mask].reshape(-1,self.dimension)
                self.radii[sidx:] = spheres[1][self.obs_link_mask].reshape(-1)
                self.center_jac[sidx:] = spheres[2][self.obs_link_mask].reshape(-1,self.dimension,self.n_params)
                self.radii_jac[sidx:] = spheres[3][self.obs_link_mask].reshape(-1,self.n_params)
            else:
                self.centers[sidx:] = spheres[0].swapaxes(1,2).reshape(-1,self.dimension)
                self.radii[sidx:] = spheres[1].swapaxes(1,2).reshape(-1)
                self.center_jac[sidx:] = spheres[2].swapaxes(1,2).reshape(-1,self.dimension,self.n_params)
                self.radii_jac[sidx:] = spheres[3].swapaxes(1,2).reshape(-1,self.n_params)


        # Do what you need with the centers and radii
        # NN(centers) - r > 0
        # D_NN(c(k)) -> D_c(NN) * D_k(c) - D_k(r)
        # D_c(NN) should have shape (n_spheres, 3, 1)
        # D_k(c) has shape (n_spheres, 3, n_params)
        # D_k(r) has shape (n_spheres, n_params)
        # dist, dist_jac = self.NN_fun(self.centers, self.obs_tuple)
    
        # batched_centers = spheres[0].swapaxes(0,1).reshape(self.n_time, -1, 3)
        # batched_radii = spheres[1].swapaxes(0,1).reshape(self.n_time, -1)
        # batched_center_jac = spheres[2].swapaxes(0,1).reshape(self.n_time, -1, 3, 7)
        # batched_radii_jac = spheres[3].swapaxes(0,1).reshape(self.n_time, -1, 7)
        centers = self.centers.view(-1, self.n_time, self.dimension)
        center_jac = self.center_jac.view(-1, self.n_time, self.dimension, self.n_params)
        radii = self.radii.view(-1, self.n_time)
        radii_jac = self.radii_jac.view(-1, self.n_time, self.n_params)

        # (-1, n_time, 3/1, ...)
        # torch.cuda.synchronize()
        # tic=time.perf_counter()
        integral, grads = SplanningConstraints._integrator(
            (self.centers, self.radii),
            (
                self.splats.mu_prime[self.gaussian_mask],
                self.splats.inv_eigv[self.gaussian_mask],
                self.splats.rotmats[self.gaussian_mask],
                self.splats.opacities[self.gaussian_mask] * self.splats.norm_coeffs[self.gaussian_mask],
            ),
            return_gradients=True)
        integral = integral.view(-1, self.n_time)
        # torch.cuda.synchronize()
        # toc=time.perf_counter()
        # print("Integration Time:", toc-tic)
        sphere_collision_probabilities = 1./self.alpha * (1 - torch.exp(-(1 / (4*torch.pi))*integral))

        # N_time x N_spheres
        dcons_dintegral = torch.exp(-integral/(4*torch.pi)) / (4*self.alpha*torch.pi)
        
        # N_time x N_spheres x (3/1)
        dintegral_dcenter, dintegral_drad = grads[0].view(-1, self.n_time, self.dimension), grads[1].view(-1, self.n_time)
        dcons_dcenter = dcons_dintegral[...,None] * dintegral_dcenter
        dcons_drad    = dcons_dintegral * dintegral_drad
    
        constraint_val: torch.Tensor = sphere_collision_probabilities 
        dcons_dk: torch.Tensor       = (dcons_dcenter[..., None,:] @ center_jac).squeeze() \
                                    + (dcons_drad[..., None] * radii_jac)
        
        if self.constraint_aggregation_mode == ConstraintAggregationMode.NONE:
            constraint_val_out = constraint_val.flatten() - self.constraint_rhs
            dcons_dk_out = dcons_dk.flatten(end_dim=1)
        elif self.constraint_aggregation_mode == ConstraintAggregationMode.TIME_INTERVALS:
            constraint_val_out = constraint_val.sum(0) - self.constraint_rhs
            dcons_dk_out = dcons_dk.sum(0)
        elif self.constraint_aggregation_mode == ConstraintAggregationMode.ALL:
            constraint_val_out = constraint_val.sum() - self.constraint_rhs
            dcons_dk_out = dcons_dk.sum([0,1])
        else:
            raise RuntimeError(f"Unknown constraint aggregation mode {self.constraint_aggregation_mode}")
                  
        n_splat_cons = constraint_val_out.nelement()

        con_dists_out = Cons_out[:n_splat_cons]
                
        con_dists_out.copy_(constraint_val_out)
        
        cons_dists_jac_out = Jac_out[:n_splat_cons]
        cons_dists_jac_out.copy_(dcons_dk_out)

        ### Add self collision if enabled
        if self.n_self_collision > 0:
            self_collision_dists = Cons_out[n_splat_cons:]
            self_collision_jacs = Jac_out[n_splat_cons:]
            joint_centers_all = centers[:self.n_joints]
            joint_radii_all = radii[:self.n_joints]
            joint_jacs_all = center_jac[:self.n_joints]
            self.__self_collision(spheres, joint_centers_all, joint_radii_all, joint_jacs_all, self_collision_dists, self_collision_jacs)
        return Cons_out, Jac_out

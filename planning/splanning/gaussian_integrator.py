from typing import Tuple
from torch import Tensor

from enum import Enum
import torch


class INTEGRATION_NORM(str, Enum):
    INF = 'inf',


@torch.no_grad()
def integrate_gaussians_spheres(
        spheres: Tuple[Tensor, Tensor],
        gaussians: Tuple[Tensor, Tensor, Tensor, Tensor],
        norm: INTEGRATION_NORM = INTEGRATION_NORM.INF,
        return_gradients: bool = False):
    
    if norm != INTEGRATION_NORM.INF:
        raise NotImplementedError
    
    # ensure output has correct shape
    batch_shape_sphere = spheres[0].shape[:-1]

    sphere_c_in = spheres[0]
    sphere_r_in = spheres[1]
    

    sphere_c = sphere_c_in.reshape(-1, 3)
    rots = gaussians[2].reshape(-1, 3)
    sphere_c_prime = (sphere_c_in@rots.T).reshape(sphere_c.shape[0], -1, 3)
    
    sphere_r = sphere_r_in.reshape(-1, 1, 1)
    mu_prime = gaussians[0].reshape(1, -1, 3)
    inv_cov = gaussians[1].reshape(1, -1, 3)

    # norm constants don't need to worry about last dim
    opacity = gaussians[3].reshape(1, -1)

    # common terms
    sqrt_2_inv_cov = torch.sqrt(inv_cov/2)                  # (n_gaussians, 1, 3)
    sqrt_pi_2_inv_cov = torch.sqrt(torch.pi/(2*inv_cov))    # (n_gaussians, 1, 3)
    delta_mu_center = mu_prime - sphere_c_prime             # (n_time, n_spheres, n_gaussians, 3)

    # compute the integration for the infinity norm
    # https://www.wolframalpha.com/input?i=integral+exp%28-0.5+*+%28%28x+-+mu%29+%2F+sqrt%28s%29%29%5E2%29+from+-r+to+r
    in1 = sqrt_2_inv_cov * (delta_mu_center + sphere_r)
    in2 = sqrt_2_inv_cov * (delta_mu_center - sphere_r)

    expanded_res = sqrt_pi_2_inv_cov * (torch.special.erf(in1) - torch.special.erf(in2)) # (n_spheres, n_gaussians, 3)
    
    # reduce to each sphere
    res = expanded_res.prod(dim=-1)
    
    res = (opacity*res).sum(dim=-1).reshape(batch_shape_sphere)
    
    # Add autograd if requested
    if return_gradients:    
        # compute gradients
        grad_alpha1 = torch.exp((-inv_cov/2) * (delta_mu_center + sphere_r)**2)
        grad_alpha2 = torch.exp((-inv_cov/2) * (delta_mu_center - sphere_r)**2)
        T1 = (grad_alpha2 - grad_alpha1)
        T2 = (grad_alpha1 + grad_alpha2)
        coeff1 = expanded_res[..., 1] * expanded_res[..., 2] * opacity
        coeff2 = expanded_res[..., 2] * expanded_res[..., 0] * opacity
        coeff3 = expanded_res[..., 0] * expanded_res[..., 1] * opacity
        sphere_c_grad = torch.sum(
                (coeff1 * T1[..., 0]).unsqueeze(-1) * gaussians[2][:,0,:] \
                + (coeff2 * T1[..., 1]).unsqueeze(-1) * gaussians[2][:,1,:] \
                + (coeff3 * T1[..., 2]).unsqueeze(-1) * gaussians[2][:,2,:], dim=1)
        sphere_r_grad = torch.sum(
                coeff1 * T2[..., 0] \
                + coeff2 * T2[..., 1] \
                + coeff3 * T2[..., 2], dim=1)
        
        sphere_c_grad = sphere_c_grad.reshape(batch_shape_sphere + (3,))
        sphere_r_grad = sphere_r_grad.reshape(batch_shape_sphere)

        return res, (sphere_c_grad, sphere_r_grad)
    return res
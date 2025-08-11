
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from planning.splanning.gaussian_integrator import integrate_gaussians_spheres
from planning.splanning.splat_tools import SplatLoader
from collision_comparisons.splat_nav.intersection_utils import compute_intersection_linear_motion


class SplatConstraints:
    _integrator = None
    def __init__(self, device, dtype):
        self.device = device
        self.dtype = dtype
        if SplatConstraints._integrator is None:
            try:
                SplatConstraints._integrator = torch.compile(integrate_gaussians_spheres, dynamic=True, options={'shape_padding': True, 'epilogue_fusion': True, 'max_autotune': True})
            except:
                print("Failed to compile the integration. Falling back to the super slow mode")
                SplatConstraints._integrator = integrate_gaussians_spheres
    
    def setup_splat(
            self,
            splat_path,
            radial_culling_radius = None,
            radial_culling_sigma_mag = None,
            sigma_magnification = None,
        ):
        splats = SplatLoader(
            filename=splat_path,
            device=self.device,
            dtype=self.dtype,
        )

        if radial_culling_radius is not None:
            kwargs = {}
            if radial_culling_sigma_mag is not None:
                kwargs['cutoff_sigma_mag'] = radial_culling_sigma_mag
            splats.radial_culling([0,0,0], radial_culling_radius, copy=False, **kwargs)
        if sigma_magnification is not None:
            splats.precompute_errors_for_sigma_mag(sigma_magnification)
        self.splats = splats
        return splats
    
    def splanning_gsplats_constraint(self, centers, radii, alphas):
        '''Integrate the gaussians for the given centers and radii. Batched across multiple alphas.
        
        Args:
            centers (torch.Tensor): The centers of the spheres. Shape (*batch_size, 3)
            radii (torch.Tensor): The radii of the spheres. Shape (*batch_size,)
            splats (SplatLoader): The splat loader object.
            alphas (torch.Tensor): The alpha values to integrate with. Shape (n_alphas)
        
        Returns:
            torch.Tensor: The collision probabilities for the spheres. Shape (n_alphas, *batch_size)
        '''
        splats = self.splats
        centers = torch.as_tensor(centers, dtype=self.dtype, device=self.device)
        radii = torch.as_tensor(radii, dtype=self.dtype, device=self.device)

        # Cull the gaussians that are far away from any sphere
        sphere_c = centers.reshape(-1, 1, 3)
        sphere_r = radii.reshape(-1, 1)
        gauss_r = splats.sigma_mag_r
        dists = torch.linalg.vector_norm(sphere_c - splats.mu, dim=-1) < gauss_r + sphere_r
        gaussian_mask = torch.any(dists, dim=0)

        # Compute the integral, returns shape (*batch_size)
        integral = SplatConstraints._integrator(
            (centers, radii),
            (
                splats.mu_prime[gaussian_mask],
                splats.inv_eigv[gaussian_mask],
                splats.rotmats[gaussian_mask],
                splats.opacities[gaussian_mask] * splats.norm_coeffs[gaussian_mask],
            ),
            return_gradients=False)

        # Commpute return probabilities over all alphas
        n_batch_dims = len(integral.shape)
        integral = integral.unsqueeze(0)
        alphas = alphas.reshape((-1,) + (1,) * n_batch_dims)
        sphere_collision_probabilities = 1./alphas * (1 - torch.exp(-(1 / (4*torch.pi))*integral))
        return sphere_collision_probabilities

    def splatnav_gsplats_constraint(self, centers, radii, sigma_levels):
        '''Integrate the gaussians for the given centers and radii. Batched across multiple sigma levels.
        
        Args:
            centers (torch.Tensor): The centers of the spheres. Shape (*batch_size, 3)
            radii (torch.Tensor): The radii of the spheres. Shape (*batch_size,)
            splats (SplatLoader): The splat loader object.
            sigma_levels (torch.Tensor): The alpha values to integrate with. Shape (n_sigmas)
        
        Returns:
            torch.Tensor: The collision boolean for the spheres. Shape (sigma_levels, *batch_size)
        '''
        splats = self.splats

        # n_spheres, n_splats, 3
        sphere_c = torch.as_tensor(centers, dtype=self.dtype, device=self.device).reshape(-1, 1, 3)
        sphere_r = torch.as_tensor(radii, dtype=self.dtype, device=self.device).reshape(-1, 1)

        # Transform for use with splatnav
        collisions = []
        from tqdm import tqdm
        for sigma in tqdm(sigma_levels, dynamic_ncols=True, desc='Splat-Nav Sigmas', position=1):
            # Mask on a per sigma level, per_sphere level
            scales = torch.sqrt(splats.eigv * sigma)
            for c, r in zip(sphere_c.reshape(-1, 3), sphere_r.reshape(-1,)):
                gaussian_mask = torch.linalg.vector_norm(c - splats.mu, dim=-1) < sigma + r
                mu = splats.mu[gaussian_mask]
                rot_mats = splats.rotmats[gaussian_mask]
                output = compute_intersection_linear_motion(
                    x0=c,
                    delta_x=torch.zeros_like(c)+1e-8,
                    R_A=rot_mats,
                    S_A=scales[gaussian_mask],
                    mu_A=mu,
                    S_B=r.item()
                )
                collisions.append((~output['is_not_intersect']).any())
                    
        collisions = torch.stack(collisions).reshape(sigma_levels.shape + centers.shape[:-1])
        return collisions

from __future__ import annotations
from typing import TYPE_CHECKING
import torch
import numpy as np
import math
from functools import lru_cache
import os
from scipy.spatial.transform import Rotation

if TYPE_CHECKING:
    from torch import Tensor, device, dtype
    from numpy import ndarray
    from typing import Union


# Default splat dat
DEFAULT_SPLAT = os.path.join(os.path.dirname(__file__), "splat.csv")

class SplatLoader:
    def __init__(
            self,
            filename: str = DEFAULT_SPLAT,
            skip_header: bool = True,
            device: device = None,
            dtype: dtype = None,
            verbose: bool = False,
            skip_step=None,
            N=None,
            data = None,
            T_world_robot: torch.Tensor = None
            ) -> None:
        """
        Loads the splat data from the given filename. The data is preprocessed
        to make it easier to use in the integration function.

        Args:
            filename (str, optional): Filename of the splat data. Defaults to splat.csv in the class folder
            skip_header(bool, optional): Whether to skip the header of the CSV file. Defaults to True.
            device (device, optional): Device to load the data on. Defaults to None.
            dtype (dtype, optional): Data type to load the data on. Defaults to None.
            verbose (bool, optional): Whether to print out progress. Defaults to False.
        """
        
        val_tensor = torch.empty(0, device=device, dtype=dtype) # Make sure device and dtype are valid
        self.device = val_tensor.device
        self.dtype = val_tensor.dtype
        self.filename = filename

        # def load_tensor(path):
        #     return next(torch.jit.load("/home/sethgi/Documents/Research/NeRF_planning/splanning_hlp/build/debug/" + path).parameters())

        # cpp_quat = load_tensor("GaussianSplat/quat.pt")
        # cpp_gaussian_transforms = load_tensor("GaussianSplat/gaussian_transforms.pt")
        # cpp_mu = load_tensor("GaussianSplat/mu.pt")
        # cpp_rotmats_in = load_tensor("GaussianSplat/rotmats_in.pt")
        # cpp_rotmats = load_tensor("GaussianSplat/rotmats.pt")
        # cpp_T_world_robot = load_tensor("GaussianSplat/T_world_robot.pt")
        # cpp_norm_coeffs = load_tensor("GaussianSplat/norm_coeffs.pt")



        # load data
        if data is None:
            if self.filename.endswith('.csv'):
                data = np.loadtxt(self.filename, delimiter=',', skiprows=1 if skip_header else 0)
            elif self.filename.endswith('.ply'):
                from plyfile import PlyData
                import warnings
                warnings.warn("Loading PLY file directly. This doesn't handle normalization " +
                              "constants which may be present in metadata for the normalized 3DGS.")
                with open(self.filename, 'rb') as f:
                    plydata = PlyData.read(f)
                xyz = ['x', 'y', 'z']
                xyz = np.asarray([plydata['vertex'][n] for n in xyz])
                scaling = ['scale_0', 'scale_1', 'scale_2']
                scaling = np.asarray([plydata['vertex'][n] for n in scaling])
                scaling = np.exp(scaling)
                rot = ['rot_0', 'rot_1', 'rot_2', 'rot_3']
                rot = np.asarray([plydata['vertex'][n] for n in rot])
                norm_rot = np.linalg.norm(rot, axis=0)
                norm_rot[norm_rot==0] = 1
                rot = (rot / norm_rot)
                opacity = np.asarray([plydata['vertex']['opacity']])
                opacity = 1 / (1 + np.exp(-opacity))
                data = np.concatenate((xyz, scaling, rot, opacity)).T
                del plydata
            data = torch.from_numpy(data).to(self.device, dtype=self.dtype)

        if verbose:
            print("Loaded splat data. Preprocessing...")
        
        self._mu = data[:, :3].contiguous()
        self._eigv = data[:, 3:6].contiguous()**2 
        self._quat = data[:, 6:10].contiguous()


        self._sigma_r = torch.sqrt(torch.max(self._eigv, dim=-1).values)

        self._rotmats = build_rotation(self._quat)

        if T_world_robot is not None:
            T_world_robot = T_world_robot.to(self.device)

            points_world = self._mu.clone()
            points_homo = torch.hstack((points_world, torch.ones_like(points_world[:, 0:1]))).T
            N = self._mu.shape[0]
            gaussian_transforms = torch.zeros(N,4,4).to(self._mu)
            gaussian_transforms[:, :3, :3] = self._rotmats
            gaussian_transforms[:, :3, 3] = self._mu
            gaussian_transforms[:, 3, 3] = 1.
            

            gaussian_transforms = T_world_robot.inverse().to(gaussian_transforms) @ gaussian_transforms

            self._rotmats = gaussian_transforms[:, :3, :3]
            self._mu = gaussian_transforms[:, :3, 3]

            # TODO: optimize this
            rotmats_np = self._rotmats.cpu().numpy()
            quats = Rotation.from_matrix(rotmats_np).as_quat()
            quats = torch.from_numpy(quats)
            quats = quats[:, [3,0,1,2]]
            self._quat = quats.to(self._quat)

        

        self._mu_prime = torch.bmm(self._rotmats, self._mu.unsqueeze(-1)).squeeze(-1)
        
        self._covariances = torch.bmm(torch.bmm(self._rotmats, torch.diag_embed(self._eigv)), self._rotmats.transpose(1,2))
        self._inv_covariances = torch.bmm(torch.bmm(self._rotmats, torch.diag_embed(1./self._eigv)), self._rotmats.transpose(1,2))
        self._inv_eigv = 1./self._eigv
        self._opacities = data[:, 10].contiguous()
        self._inv_norm_coeffs = math.sqrt((2*math.pi)**3) * torch.sqrt(torch.det(self._covariances))
        self._norm_coeffs = 1./self._inv_norm_coeffs

        valid_mask = ~self._norm_coeffs.isnan()
        self._mu = self._mu[valid_mask]
        self._eigv = self._eigv[valid_mask]
        self._quat = self._quat[valid_mask]
        self._sigma_r = self._sigma_r[valid_mask]
        self._rotmats = self._rotmats[valid_mask]
        self._mu_prime = self._mu_prime[valid_mask]
        self._covariances = self._covariances[valid_mask]
        self._inv_covariances = self._inv_covariances[valid_mask]
        self._inv_eigv = self._inv_eigv[valid_mask]
        self._opacities = self._opacities[valid_mask]
        self._inv_norm_coeffs = self._inv_norm_coeffs[valid_mask]
        self._norm_coeffs = self._norm_coeffs[valid_mask]

        self._pmf_error = None
        self._splat_errors = None
        self._sigma_magnification = None
        self._sigma_mag_r = None
        self._cull_params = None
        
        if skip_step is not None:
            self._mu = self._mu[::skip_step]
            self._eigv = self._eigv[::skip_step]
            self._quat = self._quat[::skip_step]
            self._sigma_r = self._sigma_r[::skip_step]
            self._rotmats = self._rotmats[::skip_step]
            self._mu_prime = self._mu_prime[::skip_step]
            self._covariances = self._covariances[::skip_step]
            self._inv_covariances = self._inv_covariances[::skip_step]
            self._inv_eigv = self._inv_eigv[::skip_step]
            self._opacities = self._opacities[::skip_step]
            self._inv_norm_coeffs = self._inv_norm_coeffs[::skip_step]
            self._norm_coeffs = self._norm_coeffs[::skip_step]
        
        if N is not None:
            self._mu = self._mu[:N]
            self._eigv = self._eigv[:N]
            self._quat = self._quat[:N]
            self._sigma_r = self._sigma_r[:N]
            self._rotmats = self._rotmats[:N]
            self._mu_prime = self._mu_prime[:N]
            self._covariances = self._covariances[:N]
            self._inv_covariances = self._inv_covariances[:N]
            self._inv_eigv = self._inv_eigv[:N]
            self._opacities = self._opacities[:N]
            self._inv_norm_coeffs = self._inv_norm_coeffs[:N]
            self._norm_coeffs = self._norm_coeffs[:N]

        if verbose:
            print("Done preprocessing splat data.")

    def precompute_errors_for_sigma_mag(self, sigma: float):
        """
        Precomputes the error of the PMF integration for a given sigma magnification factor.

        Args:
            sigma (float): The sigma magnification factor.
        """
        self._sigma_magnification = sigma
        self._pmf_error = error_gaussian_3d_sigma(sigma)
        self._splat_errors = self._opacities * self._inv_norm_coeffs * self._pmf_error
        self._sigma_mag_r = self._sigma_r * sigma

    def radial_culling(
            self,
            cull_centers: Union[ndarray, Tensor],
            cull_radii: Union[ndarray, Tensor, float],
            cutoff_sigma_mag: Union[ndarray, Tensor, float] = 8.5,
            base_radius = None,
            copy: bool = True,
            ) -> SplatLoader:
    
        cull_centers = torch.as_tensor(cull_centers, device=self.device, dtype=self.dtype).reshape(-1, 1, 3)
        cull_radii = torch.as_tensor(cull_radii, device=self.device, dtype=self.dtype).reshape(-1, 1)
        cutoff_sigma_mag = torch.as_tensor(cutoff_sigma_mag, device=self.device, dtype=self.dtype).expand_as(cull_radii)
        cull_params = (cull_centers, cull_radii, cutoff_sigma_mag)

        cutoff_splat_r = self._sigma_r * cutoff_sigma_mag

        if copy:
            import copy
            new_splat = copy.copy(self)
        else:
            new_splat = self
        
        dist_mask = torch.linalg.vector_norm(cull_centers - self.mu, dim=-1) < cutoff_splat_r + cull_radii
        dist_mask = dist_mask.any(dim=0)
        if base_radius is not None:
            base_mask = ~torch.logical_and(self.mu[:,0].abs() < base_radius, self.mu[:,1].abs() < base_radius) # within base
            dist_mask = torch.logical_and(dist_mask, base_mask)

        for attr in dir(self):
            if attr.startswith("__"):
                continue
            if not attr.startswith("_"):
                continue
            val = getattr(self, attr)
            if isinstance(val, torch.Tensor):
                setattr(new_splat, attr, val[dist_mask].contiguous())
        new_splat._cull_params = cull_params
        return new_splat

    def to(self, device=None, dtype=None):
        """
        Moves the data to the given device and data type.

        Args:
            device (device, optional): Device to move the data to. Defaults to None.
            dtype (dtype, optional): Data type to move the data to. Defaults to None.
        """
        if device is None:
            device = self.device
        if dtype is None:
            dtype = self.dtype
        if device == self.device and dtype == self.dtype:
            return self
        import copy
        new_splat = copy.copy(self)
        new_splat.device = device
        new_splat.dtype = dtype
        for attr in dir(self):
            if attr.startswith("__"):
                continue
            if not attr.startswith("_"):
                continue
            val = getattr(self, attr)
            if isinstance(val, torch.Tensor):
                setattr(new_splat, attr, val.to(device, dtype))
        return new_splat

    @property
    def mu(self):
        """The mean of the splat data. Shape (n, 3)"""
        return self._mu
    
    @property
    def eigv(self):
        """The eigenvalues of the splat data. Shape (n, 3)"""
        return self._eigv
    
    @property
    def quat(self):
        """The quaternion corresponding to the eigenvalues. Shape (n, 4)"""
        return self._quat
    
    @property
    def rotmats(self):
        """The rotation matrices corresponding to the eigenvalues. Shape (n, 3, 3)"""
        return self._rotmats
    
    @property
    def mu_prime(self):
        """The rotated mean corresponding to the eigenvalues. Shape (n, 3)"""
        return self._mu_prime
    
    @property
    def covariances(self):
        """The unrotated covariance matrices for the splat data. Shape (n, 3, 3)"""
        return self._covariances
    
    @property
    def inv_covariances(self):
        """The unrotated inverse covariance matrices for the splat data. Shape (n, 3, 3)"""
        return self._inv_covariances
    
    @property
    def inv_eigv(self):
        """The inverse eigenvalues of the splat data. Shape (n, 3)"""
        return self._inv_eigv
    
    @property
    def opacities(self):
        """The opacities of the splat data. Shape (n,)"""
        return self._opacities
    
    @property
    def inv_norm_coeffs(self):
        """The inverse normalization coefficients of the splat data. Shape (n,)"""
        return self._inv_norm_coeffs
    
    @property
    def norm_coeffs(self):
        """The normalization coefficients of the splat data. Shape (n,)"""
        return self._norm_coeffs
    
    @property
    def sigma_magnification(self):
        """The sigma magnification factor for the PMF integration. Shape (n,)"""
        return self._sigma_magnification

    @property
    def pmf_error(self):
        """The error of the PMF integration for a given sigma magnification factor. Shape (n,)"""
        return self._pmf_error
    
    @property
    def splat_errors(self):
        """The error of the integrated splats for a given sigma magnification factor. Shape (n,)"""
        return self._splat_errors
    
    @property
    def sigma_r(self):
        """The largest radius of the splat data corresponding to sigma. Shape (n,)"""
        return self._sigma_r
    
    @property
    def sigma_mag_r(self):
        """The largest radius of the splat data corresponding to sigma magnification. Shape (n,)"""
        return self._sigma_mag_r
    
    def __repr__(self):
        return f"SplatLoader(filename={self.filename}, device={self.device}, dtype={self.dtype})"


def build_rotation(quat: Tensor) -> Tensor:
    """
    Converts a batched quaternion to a rotation matrix. Only works for 1 batch dimension.

    Args:
        quat (Tensor): Batched quaternion of shape (batch_size, 4)

    Returns:
        Tensor: Batched rotation matrix of shape (batch_size, 3, 3)
    """
    norm = torch.sqrt(quat[:,0]*quat[:,0] + quat[:,1]*quat[:,1] + quat[:,2]*quat[:,2] + quat[:,3]*quat[:,3])

    q = quat / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device=quat.device, dtype=quat.dtype)

    w = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - w*z)
    R[:, 0, 2] = 2 * (x*z + w*y)
    R[:, 1, 0] = 2 * (x*y + w*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - w*x)
    R[:, 2, 0] = 2 * (x*z - w*y)
    R[:, 2, 1] = 2 * (y*z + w*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R


# The table is generated from the wolframlanguagescript in the same folder
# called compute_gaussian_error.wls. The valid range of the table is from
# 2 to 8.5. Larger than 8.5, the numerical integration fails. Values are
# interpolated from the table, with the table values being 0.02 apart.

@lru_cache()
def error_gaussian_3d_sigma(sigma_value):
    if sigma_value < 2 or sigma_value > 8.5:
        raise ValueError("sigma_value must be between 2 and 8.5")
    
    # Load the CSV file
    data = np.loadtxt(os.path.join(os.path.dirname(__file__), 'gaussian_table.csv'), delimiter=',')

    # Extract the first and second columns
    x = data[:, 0]
    y = data[:, 1]

    # Interpolate the value from the second column based on the first column
    error = np.interp(sigma_value, x, y)

    return error


from .nerf import NeRFWrapper
import numpy as np
from .purr import Catnips
import torch

class CatnipsConstraint:
    def __init__(self, nerf_wrapper: NeRFWrapper, catnips_configs, sphere_radius):
        self.nerf_wrapper = nerf_wrapper
        grid = np.array(catnips_configs["grid"])

        inscribed_side_length = 2 * sphere_radius * np.sqrt(3) / 3
        agent_body = np.array(
            [[-inscribed_side_length / 2, inscribed_side_length / 2]] * 3
        ) * self.nerf_wrapper.scale

        cfg = {
            'grid': grid,
            'agent_body': agent_body,
            'sigma': catnips_configs["sigma"],
            'Aaux': catnips_configs["Aaux"],
            'dt': catnips_configs["dt"],
            'Vmax': catnips_configs["Vmax"],
            'gamma': catnips_configs["gamma"],
            'discretization': catnips_configs["discretization"],
            'density_factor': 1,
            'get_density': self.nerf_wrapper.get_density
        }

        self._catnips = Catnips(cfg)
        self._catnips.load_purr()
        self._catnips.create_purr()
        # self._catnips.save_purr(
        #     "test.ply", self.nerf_wrapper.transform.detach().cpu().numpy(), self.nerf_wrapper.scale)

        self.grid_points = self._catnips.conv_centers        
        self.cell_sizes = self.grid_points[1, 1, 1] - self.grid_points[0, 0, 0]
        self.grid_occupied = ~self._catnips.purr

    def constraint(self, centers):
        # Based on the catnips code: corridor.init_path.PathInit.get_indices()
        original_shape = centers.shape
        centers = torch.from_numpy(centers.reshape(-1, 3)).cuda().float()
        points = self.nerf_wrapper.data_frame_to_ns_frame(centers).cpu().numpy()
        min_bound = self.grid_points[0, 0, 0] - self.cell_sizes / 2
        transformed_pt = points - min_bound
        indices = (transformed_pt / self.cell_sizes).astype(np.int32)

        clipped_indices = np.clip(indices, 0, np.array(self.grid_occupied.shape) - 1)

        if np.any(indices != clipped_indices):
            print('Point(s) outside grid bounds projected to nearest side. May cause unintended behavior.')

        occupied = self.grid_occupied[clipped_indices[:, 0],clipped_indices[:, 1],clipped_indices[:, 2]]
        
        return occupied.reshape(original_shape[:2])
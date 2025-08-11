from __future__ import annotations
import torch
from .jrs_trig.process_jrs_trig import process_batch_JRS_trig as _process_batch_JRS_trig
from .jrs_trig.load_jrs_trig import JRSInfo
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Tuple, Union
    from numpy import ndarray
    from torch import Tensor
    Array = Union[Tensor, ndarray]

class OfflineJRS:
    def __init__(
        self,
        jrs_path = None,
        device: torch.device = 'cpu',
        dtype: torch.dtype = torch.float,
        ):

        """ Wrapper for preloading and processing JRS tensors """
        self.jrs_info = JRSInfo(jrs_path, dtype, device)
        
        self.jrs_tensor = self.jrs_info.jrs_tensor
        self.g_ka = self.jrs_info.g_ka
        self.device = device
        self.dtype = dtype

    def __call__(
        self,
        qpos: Array,
        qvel: Array,
        joint_axes: Array,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Returns the JRS and the corresponding rotatotopes for a given configuration and velocity
        Args:
            qpos (torch.Tensor): The configuration of the robot
            qvel (torch.Tensor): The velocity of the robot
            joint_axes (torch.Tensor): The joint axes of the robot

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The JRS and the corresponding rotatotopes
        """
        qpos = torch.as_tensor(qpos, dtype=self.dtype, device=self.device)
        qvel = torch.as_tensor(qvel, dtype=self.dtype, device=self.device)
        joint_axes = torch.as_tensor(joint_axes, dtype=self.dtype, device=self.device)
        return _process_batch_JRS_trig(self.jrs_tensor, qpos, qvel, joint_axes)

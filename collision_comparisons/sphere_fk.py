import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import zonopyrobots as zpr

from forward_occupancy.SO import sphere_occupancy, make_spheres
from collision_comparisons.geometry import rot_mat

class SphereFK:
    '''Simple class to generate spheres for a robot using forward kinematics like SPARROWS and Splanning.

    Args:
        robot (zpr.ZonoArmRobot): The robot to generate spheres for.
        joint_radius_override (dict, optional): A dictionary to override the joint radii. Defaults to {}.
        n_spheres_per_link (int, optional): The number of spheres per link. Defaults to 5.
    '''
    def __init__(
            self,
            robot: zpr.ZonoArmRobot,
            joint_radius_override: dict = {},
            n_spheres_per_link: int = 5,
            specified_links: list = None,
            include_joints: bool = True,
        ):
        self.robot = robot
        self.joint_radius_override = joint_radius_override
        self.device = robot.device
        self.dtype = robot.dtype
        self.spheres_per_link = n_spheres_per_link
        self.specified_links = specified_links
        self.include_joints = include_joints
        self.__preprocess_link_pairs()

    def __preprocess_link_pairs(self):
        ## Preprocess some constraint constants
        joint_occ, link_joint_pairs, _ = sphere_occupancy({}, self.robot, 2, joint_radius_override=self.joint_radius_override)
        # Flatten out all link joint pairs so we can just process that
        joint_pairs = []
        for pairs in link_joint_pairs.values():
            joint_pairs.extend(pairs)
        joints_idxs = {name: i for i, name in enumerate(joint_occ.keys())}
        # convert to indices
        p_idx = torch.empty((2, len(joint_pairs)), dtype=int, device=self.device)
        for i, (joint1, joint2) in enumerate(joint_pairs):
            p_idx[0][i] = joints_idxs[joint1]
            p_idx[1][i] = joints_idxs[joint2]
        self.p_idx = p_idx.to(device=self.device)

    def generate_spheres(self, joint_angles):
        '''Takes in the joint angles and returns the centers and radii of the spheres that represent the robot joints and links.
        
        Implemented in a batchable manner, and returns joint spheres and link spheres. n_spheres_per_link is the number of spheres per link
        as defined in the constructor.
        
        Args:
            joint_angles (torch.Tensor): The joint angles of the robot. Shape (*batch_shape, n_joints)
        
        Returns:
            joint_centers (torch.Tensor): The centers of the joint spheres. Shape (n_joints, *batch_shape, 3)
            joint_radii (torch.Tensor): The radii of the joint spheres. Shape (n_joints, *batch_shape)
            centers (torch.Tensor): The centers of the link spheres. Shape (n_links, *batch_shape, n_spheres_per_link, 3)
            radii (torch.Tensor): The radii of the link spheres. Shape (n_links, *batch_shape, n_spheres_per_link)
        '''
        joint_angles = torch.as_tensor(joint_angles, dtype=self.dtype, device=self.device)
        rotmats = [rot_mat(joint_angles[..., i], self.robot.joint_axis[i]) for i in range(self.robot.dof)]
        joint_occ, _, _ = sphere_occupancy(rotmats, self.robot, 2, joint_radius_override=self.joint_radius_override)
        
        # We can just take the pz centers because all inputs aren't sets
        # If batching, the current sphere_occupancy returns a pz for the first joint,
        # a batchpz for the later ones, so account for that with expand_as
        joint_centers = [pz.c for pz, _ in joint_occ.values()]
        joint_centers = torch.stack([c.expand_as(joint_centers[-1]) for c in joint_centers])
        joint_radii = [r for _, r in joint_occ.values()]
        joint_radii = torch.stack([r.expand_as(joint_radii[-1]) for r in joint_radii])

        # Fill in the link spheres
        joint_centers_check = joint_centers[self.p_idx]
        joint_radii_check = joint_radii[self.p_idx]
        centers, radii = make_spheres(joint_centers_check[0], joint_centers_check[1], joint_radii_check[0], joint_radii_check[1], n_spheres=self.spheres_per_link)

        return joint_centers, joint_radii, centers, radii
    
    def generate_select_spheres(self, joint_angles):
        '''Takes in the joint angles and returns the centers and radii of the spheres that represent the robot joints and links.

        Implemented in a batchable manner, and returns joint spheres and link spheres. n_spheres_per_link is the number of spheres per link
        as defined in the constructor. This function only returns the spheres for the specified links and joints if specified_links is not None.
        It also permutes the dimensions to put the batch dimension first.

        Args:
            joint_angles (torch.Tensor): The joint angles of the robot. Shape (*batch_shape, n_joints)

        Returns:
            centers (torch.Tensor): The centers of the link spheres. Shape (*batch_shape, n_robot_spheres, 3)
            radii (torch.Tensor): The radii of the link spheres. Shape (*batch_shape, n_robot_spheres)
        
        See Also:
            generate_spheres: The function that generates all spheres which this function filters, consolidates, and permutes.
        '''
        if self.specified_links is None:
            filter_ = list(range(self.robot.dof))
        else:    
            filter_ = self.specified_links
        joint_centers, joint_radii, centers, radii = self.generate_spheres(joint_angles)
        link_centers = centers[filter_]
        link_radii = radii[filter_]

        consolidated_centers = []
        consolidated_radii = []
        added_joints = []
        for lc, lr, lidx in zip(link_centers, link_radii, filter_):
            if self.include_joints and lidx not in added_joints:
                consolidated_centers.append(joint_centers[lidx][...,None,:])
                consolidated_radii.append(joint_radii[lidx][...,None])
                added_joints.append(lidx)
            consolidated_centers.append(lc)
            consolidated_radii.append(lr)
            if self.include_joints:
                consolidated_centers.append(joint_centers[lidx+1][...,None,:])
                consolidated_radii.append(joint_radii[lidx+1][...,None])
                added_joints.append(lidx+1)
        
        consolidated_centers = torch.concat(consolidated_centers, dim=-2)
        consolidated_radii = torch.concat(consolidated_radii, dim=-1)
        return consolidated_centers, consolidated_radii

    def __repr__(self):
        return  f"SphereFK(robot={self.robot}, " + \
                f"joint_radius_override={self.joint_radius_override}, " + \
                f"n_spheres_per_link={self.spheres_per_link}, " + \
                f"specified_links={self.specified_links}, " + \
                f"include_joints={self.include_joints})"



if __name__=='__main__':

    test = SphereFK(robot=zpr.ZonoArmRobot(zpr.robots.urdfs.KinovaGen3, create_joint_occupancy=True), specified_links=[2,4,6])
    joint_angles = torch.tensor([[0.0, 0.1*np.pi, np.pi/2, 0.0, 0.0, 0.0, 0.0, 0.0], 
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    joint_angles = torch.stack([joint_angles, joint_angles])
    test.generate_spheres(joint_angles)
    # scene_no = 0
    # splat_file = f"/mnt/ws-frb/users/sethgi/splanning_data/experiment_data/scene_{int(scene_no)}/splats/iteration_10000.csv"

    # collision_checker = FKCollision(splat_file=splat_file)

    # # Define joint angles (in radians) for multiple samples
    # joint_angles = torch.tensor([[0.0, 0.1*np.pi, np.pi/2, 0.0, 0.0, 0.0, 0.0, 0.0], 
    #                             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
    #                             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
    #                             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
    #                             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
    #                             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    
    # # total_joint_angles = np.load('../prm_tests/joint_list.npy')
    # total_joint_angles = joint_angles
    # breakpoint()
    # total_joint_angles = torch.tensor(total_joint_angles, dtype=torch.float)
    # collision_checker.get_collision_batch(joint_angles=total_joint_angles)
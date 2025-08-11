import torch


import numpy as np
import zonopy as zp
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from forward_occupancy.JRS import OfflineJRS
from forward_occupancy.SO import sphere_occupancy, make_spheres
import cyipopt
import yaml
import time
import argparse

T_PLAN, T_FULL = 0.5, 1.0

from typing import List
from planning.splanning.splanning_constraints import SplanningConstraints, ConstraintAggregationMode
from planning.splanning.splat_tools import SplatLoader
from planning.common.base_armtd_nlp_problem import BaseArmtdNlpProblem
import zonopyrobots as zpr
from zonopyrobots import ZonoArmRobot
from planning.common.waypoints import ArmWaypoint

def load_config(filename):
    with open(filename, 'r') as file:
        config = yaml.safe_load(file)
    return config

class Splanner:
    def __init__(self,
                 robot: zpr.ZonoArmRobot,
                 zono_order: int = 2, # this appears to have been 40 before but it was ignored for 2
                 max_combs: int = 200,
                 dtype: torch.dtype = torch.float,
                 device: torch.device = torch.device('cpu'),
                 sphere_device: torch.device = torch.device('cpu'),
                 spheres_per_link: int = 5,
                 use_weighted_cost: bool = False,
                 joint_radius_override: dict = {},
                 filter_links: bool = False,
                 check_self_collisions: bool = False,
                 splats: SplatLoader = None,
                 constraint_aggregation_mode = ConstraintAggregationMode.TIME_INTERVALS,
                 constraint_beta=0.05,
                 constraint_alpha=0.05,
                 linear_solver='ma57',
                 planner_kd=0.1,
                 ignore_last_joint = False,
                 jrs_path=None,
                 ):
        
        
        self.dtype, self.device = dtype, device
        self.np_dtype = torch.empty(0,dtype=dtype).numpy().dtype
        self.constraint_aggregation_mode = constraint_aggregation_mode
        self.constraint_alpha = constraint_alpha
        self.constraint_beta = constraint_beta
        
        self.linear_solver = linear_solver
        
        self.ignore_last_joint = ignore_last_joint

        self.robot = robot
        self.PI = torch.tensor(torch.pi,dtype=self.dtype,device=self.device)
        self.JRS = OfflineJRS(dtype=self.dtype,device=self.device, jrs_path=jrs_path)
        
        self.zono_order = zono_order
        self.max_combs = max_combs
        self.combs = self._generate_combinations_upto(max_combs)

        self._setup_robot(robot)
        self.sphere_device = sphere_device
        self.spheres_per_link = spheres_per_link
        
        if splats is not None:
            self.set_splats(splats)
        
        self.joint_radius_override = joint_radius_override
        self.filter_links = filter_links
        self.check_self_collisions = check_self_collisions
        self.__preprocess_SO_constants()

        # Prepare the nlp
        self.g_ka = np.ones((self.dof),dtype=self.np_dtype) * self.JRS.g_ka
        # Radius of maximum steady state oscillation that can occur for the given g_ka
        self.osc_rad = self.g_ka*(T_PLAN**2)/8
        self.nlp_problem_obj = BaseArmtdNlpProblem(self.dof,
                                         self.g_ka,
                                         self.pos_lim, 
                                         self.vel_lim,
                                         self.continuous_joints,
                                         self.pos_lim_mask,
                                         self.dtype,
                                         T_PLAN,
                                         T_FULL,
                                         weight_joint_costs = use_weighted_cost,
                                         kd_value=planner_kd)
                                        #  ignore_last_joint=self.ignore_last_joint)

        if self.ignore_last_joint:
            print("Warning: ignore_last_joint was set but is deprecated. Argument will be ignored.")
        
        self._has_warned_splat = False

    def set_splats(self, splat: SplatLoader):
        self.SFO_constraint = SplanningConstraints(dtype=self.dtype,
                                            device=self.sphere_device,
                                            n_params=self.dof,
                                            splat_loader=splat,
                                            constraint_aggregation_mode=self.constraint_aggregation_mode,
                                            constraint_rhs=self.constraint_beta,
                                            constraint_alpha=self.constraint_alpha)


    def _setup_robot(self, robot: zpr.ZonoArmRobot):
        self.dof = robot.dof
        self.joint_axis = robot.joint_axis
        self.pos_lim = robot.np.pos_lim
        self.vel_lim = robot.np.vel_lim
        # self.vel_lim = np.clip(robot.np.vel_lim, a_min=None, a_max=self.JRS.g_ka * T_PLAN)
        # self.eff_lim = np.array(eff_lim) # Unused for now
        self.continuous_joints = robot.np.continuous_joints
        self.pos_lim_mask = robot.np.pos_lim_mask
        pass

    def _generate_combinations_upto(self, max_combs):
        return [torch.combinations(torch.arange(i,device=self.device),2) for i in range(max_combs+1)]
    
    def __preprocess_SO_constants(self):
        ## Preprocess some constraint constants
        joint_occ, link_joint_pairs, _ = sphere_occupancy({}, self.robot, self.zono_order, joint_radius_override=self.joint_radius_override)
        # Flatten out all link joint pairs so we can just process that
        joint_pairs = []
        for pairs in link_joint_pairs.values():
            joint_pairs.extend(pairs)
        joints_idxs = {name: i for i, name in enumerate(joint_occ.keys())}
        # convert to indices
        p_idx = torch.empty((2, len(joint_pairs)), dtype=int, device=self.sphere_device)
        for i, (joint1, joint2) in enumerate(joint_pairs):
            p_idx[0][i] = joints_idxs[joint1]
            p_idx[1][i] = joints_idxs[joint2]
        self.p_idx = p_idx.to(device=self.sphere_device)

        ## Preprocess self collision constants
        if self.check_self_collisions:
            self.__preprocess_self_collision_constants(joint_occ, joint_pairs, joints_idxs)
    
    def __preprocess_self_collision_constants(self, joint_occ, joint_pairs, joints_idxs):
        # we want to ignore joints that are too close
        joint_centers = torch.stack([centers.c for centers, _ in joint_occ.values()])
        joint_radii = torch.stack([radius for _, radius in joint_occ.values()])*2 # this needs a better multiplier or constant to add based on the JRS or FO
        self_collision_joint_joint = (torch.linalg.vector_norm(joint_centers.unsqueeze(0) - joint_centers.unsqueeze(1), dim=2) > joint_radii)
        # we want all link pairs that don't share joints
        self._self_collision_link_link = [None]*len(joint_pairs)
        self._self_collision_link_joint = [None]*len(joint_pairs)
        for outer_i,jp_outer in enumerate(joint_pairs):
            # get collidable joints
            collidable_vec = torch.logical_and(self_collision_joint_joint[joints_idxs[jp_outer[0]]], self_collision_joint_joint[joints_idxs[jp_outer[1]]])
            def collidable(jp_inner):
                return collidable_vec[joints_idxs[jp_inner[0]]] and collidable_vec[joints_idxs[jp_inner[1]]]
            jp_mask = torch.zeros(len(joint_pairs), dtype=torch.bool, device=self.sphere_device)
            jp_idxs = [inner_i for inner_i, jp_inner in enumerate(joint_pairs) if (jp_inner[0] not in jp_outer and jp_inner[1] not in jp_outer and collidable(jp_inner))]
            jp_mask[jp_idxs] = True
            joint_mask = torch.zeros(len(joint_occ), dtype=torch.bool, device=self.sphere_device)
            joint_idxs = torch.unique(self.p_idx[:,jp_idxs])
            joint_mask[joint_idxs] = True
            self._self_collision_link_link[outer_i] = jp_mask
            self._self_collision_link_joint[outer_i] = joint_mask
        
        # Collision pairs between links and joints
        self._self_collision_link_link = torch.stack(self._self_collision_link_link).tril() # check up to diag
        self._self_collision_link_joint = torch.stack(self._self_collision_link_joint) # check full row
        self._self_collision_joint_joint = self_collision_joint_joint.tril() # check up to diag

    def _prepare_SO_constraints(self,
                                JRS_R: zp.batchMatPolyZonotope,
                                obs_zono: zp.batchZonotope):
        ### Process the obstacles
        dist_net_time = time.perf_counter()

        # Compute hyperplanes from buffered obstacles generators
        # hyperplanes_A, hyperplanes_b = obs_zono.to(device=self.sphere_device).polytope(self.combs)
        # # Compute vertices from buffered obstacles generators
        # v1, v2 = compute_edges_from_generators(obs_zono.Z[...,0:1,:], obs_zono.Z[...,1:,:], hyperplanes_A, hyperplanes_b.unsqueeze(-1))
        # # combine to one input for the NN
        # obs_tuple = (hyperplanes_A, hyperplanes_b, v1, v2)
        obs_tuple = (None, None, None, None)

        ### get the forward occupancy
        SFO_gen_time = time.perf_counter()
        joint_occ, _, _ = sphere_occupancy(JRS_R, self.robot, self.zono_order, joint_radius_override=self.joint_radius_override)
        n_timesteps = JRS_R[0].batch_shape[0]

        # Batch construction
        centers_bpz = zp.stack([pz for pz, _ in joint_occ.values()])
        radii = torch.stack([r for _, r in joint_occ.values()])

        ## Get overapproximative spheres over all timesteps if needed
        center_check_int = centers_bpz.to_interval()
        center_check = center_check_int.center()
        rad_check = radii + torch.linalg.vector_norm(center_check_int.rad(),dim=-1)
        joint_centers_check = center_check[self.p_idx]
        joint_radii_check = rad_check[self.p_idx]
        points_check, radii_check = make_spheres(joint_centers_check[0], joint_centers_check[1], joint_radii_check[0], joint_radii_check[1], n_spheres=self.spheres_per_link)

        obs_link_mask = None

        if self.SFO_constraint.splats is not None:
            all_centers = torch.vstack((joint_centers_check.reshape(-1, 3), points_check.reshape(-1, 3)))
            all_radii = torch.hstack((joint_radii_check.reshape(-1), radii_check.reshape(-1)))

            sphere_c = all_centers.reshape(-1, 1, 3)
            sphere_r = all_radii.reshape(-1, 1)
            gauss_r = self.SFO_constraint.splats.sigma_mag_r

            dists = torch.linalg.vector_norm(sphere_c - self.SFO_constraint.splats.mu, dim=-1) < gauss_r + sphere_r
            gaussian_mask = torch.any(dists, dim=0)
        else:
            if not self._has_warned_splat:
                print("No splat detected!! Proceeding for rendering purposes only.")
                self._has_warned_splat = True
            gaussian_mask = torch.zeros(1).int()

        self_collision = None
        if self.check_self_collisions:
            link_link_mask = self._self_collision_link_link.clone()
            for link_idx, (link_spheres_c, link_spheres_r) in enumerate(zip(points_check, radii_check)):
                comp_spheres_c = points_check[self._self_collision_link_link[link_idx]] # (n_comp_links, time_idx, sphere_idx, dim)
                comp_spheres_r = radii_check[self._self_collision_link_link[link_idx]] # (n_comp_links, time_idx, sphere_idx)
                delta = link_spheres_c[None,...,None,:] - comp_spheres_c[...,None,:,:]
                dists = torch.linalg.vector_norm(delta, dim=-1)
                surf_dists = dists - link_spheres_r.unsqueeze(-1) - comp_spheres_r.unsqueeze(-2) # (n_comp_links, time_idx, sphere_idx(self), sphere_idx)
                link_link_mask[link_idx][self._self_collision_link_link[link_idx]] = torch.sum(surf_dists < 0,dim=(-1,-2,-3),dtype=bool)
            
            # Then get each link to joint distance
            link_joint_mask = self._self_collision_link_joint.clone()
            for link_idx, (link_spheres_c, link_spheres_r) in enumerate(zip(points_check, radii_check)):
                comp_spheres_c = center_check[self._self_collision_link_joint[link_idx]].unsqueeze(-2) # (n_comp_links, time_idx, 1, dim)
                comp_spheres_r = rad_check[self._self_collision_link_joint[link_idx]].unsqueeze(-1) # (n_comp_links, time_idx, 1)
                delta = link_spheres_c.unsqueeze(0) - comp_spheres_c
                dists = torch.linalg.vector_norm(delta, dim=-1)
                surf_dists = dists - comp_spheres_r - link_spheres_r # (n_comp_links, time_idx, sphere_idx)
                link_joint_mask[link_idx][self._self_collision_link_joint[link_idx]] = torch.sum(surf_dists < 0,dim=(-1,-2),dtype=bool)

            # Then get each joint to joint distance
            # joint_centers_all are (joint_idx, time_idx, dim)
            # joint_radii_all are (joint_idx, time_idx)
            joint_joint_mask = self._self_collision_joint_joint.clone()
            for joint_idx, (joint_spheres_c, joint_spheres_r) in enumerate(zip(center_check, rad_check)):
                comp_spheres_c = center_check[self._self_collision_joint_joint[joint_idx]] # (n_comp_joints, time_idx, dim)
                comp_spheres_r = rad_check[self._self_collision_joint_joint[joint_idx]] # (n_comp_joints, time_idx)
                delta = joint_spheres_c.unsqueeze(0) - comp_spheres_c
                dists = torch.linalg.vector_norm(delta, dim=-1)
                surf_dists = dists - comp_spheres_r - joint_spheres_r # (n_comp_joints, time_idx)
                joint_joint_mask[joint_idx][self._self_collision_joint_joint[joint_idx]] = torch.sum(surf_dists < 0,dim=(-1),dtype=bool)
            
            self_collision = (link_link_mask, link_joint_mask, joint_joint_mask)

        # output range
        out_g_ka = self.g_ka

        # Build the constraint
        final_time = time.perf_counter()
        self.SFO_constraint.set_params(centers_bpz, radii, self.p_idx, obs_link_mask, out_g_ka, self.spheres_per_link, n_timesteps, obs_tuple, self_collision, gaussian_mask)

        graph_time = time.perf_counter()
        out_times = {
            'graph_time': graph_time - final_time,
            'SFO_gen': final_time - SFO_gen_time,
            'distance_prep_net': SFO_gen_time - dist_net_time,
            'remaining_gaussians': torch.sum(gaussian_mask).cpu().numpy().item(),
            # 'remaining_links': torch.sum(obs_link_mask).cpu().numpy(),
        }
        return self.SFO_constraint, out_times

    def trajopt(self, qpos, qvel, waypoint, ka_0, SFO_constraint, time_limit=None, t_final_thereshold=0., debug=False):
        self.nlp_problem_obj.reset(qpos, qvel, waypoint.pos, SFO_constraint, t_final_thereshold=t_final_thereshold, qdgoal=waypoint.vel)
        n_constraints = self.nlp_problem_obj.M

        nlp = cyipopt.Problem(
        n = self.dof, # number of decision variables
        m = n_constraints,
        problem_obj=self.nlp_problem_obj,
        lb = [-1]*self.dof,
        ub = [1]*self.dof,
        cl = [-1e20]*n_constraints,
        cu = [-1e-6]*n_constraints,
        )

        #nlp.add_option('hessian_approximation', 'exact')
        nlp.add_option('sb', 'yes') # Silent Banner
        if not debug:
            nlp.add_option('print_level', 0)
        nlp.add_option('tol', 1e-5)
        nlp.add_option('linear_solver', self.linear_solver)

        if time_limit is not None:
            nlp.add_option('max_wall_time', max(time_limit, 0.01))

        if ka_0 is None:
            ka_0 = np.zeros(self.dof, dtype=np.float32)
        k_opt, self.info = nlp.solve(ka_0)          
        self.final_cost = self.info['obj_val'] if self.info['status'] == 0 else None      
        return SFO_constraint.g_ka * k_opt, self.info['status'], self.nlp_problem_obj.constraint_times
        
    def plan(self,qpos, qvel, waypoint, obs, ka_0 = None, time_limit=None, t_final_thereshold=0., debug=False, prepare_only=False):
        # prepare the JRS
        JRS_process_time_start = time.perf_counter()
        _, JRS_R = self.JRS(qpos, qvel, self.joint_axis)
        JRS_process_time = time.perf_counter() - JRS_process_time_start

        # Create obs zonotopes
        if obs is not None:
            obs_Z = torch.cat((
                torch.as_tensor(obs[0], dtype=self.dtype, device=self.device).unsqueeze(-2),
                torch.diag_embed(torch.as_tensor(obs[1], dtype=self.dtype, device=self.device))/2.
                ), dim=-2)
            obs_zono = zp.batchZonotope(obs_Z)
        else:
            obs_zono = None
            
        # Compute FO
        SFO_constraint, SFO_times = self._prepare_SO_constraints(JRS_R, obs_zono)
        if prepare_only:
            return None
        
        preparation_time = time.perf_counter() - JRS_process_time_start
        if time_limit is not None:
            time_limit -= preparation_time
            
        trajopt_time = time.perf_counter()
        k_opt, flag, constraint_times = self.trajopt(qpos, qvel, waypoint, ka_0, 
                                SFO_constraint, time_limit=time_limit, 
                                t_final_thereshold=t_final_thereshold,
                                debug=debug)
        trajopt_time = time.perf_counter() - trajopt_time
        
        timing_stats = {
            'cost': self.final_cost,
            'nlp': trajopt_time, 
            'total_prepartion_time': preparation_time,
            'JRS_process_time': JRS_process_time,
            'constraint_times': constraint_times,
            'num_constraint_evaluations': self.nlp_problem_obj.num_constraint_evaluations,
            'num_jacobian_evaluations': self.nlp_problem_obj.num_jacobian_evaluations,
        }
        timing_stats.update(SFO_times)
        
        return k_opt, flag, timing_stats

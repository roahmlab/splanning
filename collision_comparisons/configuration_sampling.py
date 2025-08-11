from environments.urdf_obstacle import KinematicUrdfWithObstacles
from collision_comparisons.sphere_fk import SphereFK
import zonopyrobots as zpr
import pandas as pd
import pickle
import numpy as np
import torch
import os
import lzma

from util import sampling_config_generator


def lazy_except_opener(file):
    try:
        with open(file, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        with lzma.open(file + ".xz", 'rb') as f:
            return pickle.load(f)


class GroundtruthConfigurationSampler:
    def __init__(
            self,
            sphere_fk: SphereFK,
            override_name: str | None = None,
            override_waypoint_csv: str | None = None,
            joint_sampling: dict = {},
            sample_stratification: dict = {},
            computed_sets: dict = {},
            max_sample_attempts: int = 1000,
            sample_size: int = 1000,
            compress: bool = False
        ):

        self.sphere_fk = sphere_fk
        self.override_name = override_name
        self.override_waypoint_csv = override_waypoint_csv
        if override_name is not None and override_waypoint_csv is not None:
            raise ValueError("Cannot specify both override_name and override_waypoint_csv.")
        if override_name is None:
            self.joint_sampling = self._process_joint_sampling(sphere_fk.robot, joint_sampling)
            self.sample_stratification = self._validate_stratification(sample_stratification)
            self.computed_sets = computed_sets
            self.max_sample_attempts = max_sample_attempts
            self.sample_size = sample_size
            self.compress = compress
            if len(computed_sets) == 0:
                raise ValueError("No computed sets specified. Must specify at least one computed set.")
        self.state = None
    
    def sample_load_environment(self, scene_file, exp_datapath: str = None, prefix: str = None, run_date: str = None):
        # If we have an override name, then that's what we need to load
        if self.override_name is not None:
            if exp_datapath is None:
                raise ValueError("Must specify datapath to load override name.")
            self.state = lazy_except_opener(os.path.join(exp_datapath, f"{self.override_name}.pkl"))
            return

        # Get persistent file location
        save_name = "samples_gt"
        if prefix is not None:
            save_name = f"{prefix}_{save_name}"
        if run_date is not None:
            save_name = f"{save_name}_{run_date}"
        
        # Check if the file exists. If it does, then load it
        save_file = None
        if exp_datapath is not None:
            save_file = os.path.join(exp_datapath, f"{save_name}.pkl")
            if os.path.exists(save_file) or os.path.exists(save_file + '.xz'):
                self.state = lazy_except_opener(save_file)
                return

        # Otherwise open the scene and process
        opener = open
        if scene_file.endswith('.xz'):
            opener = lzma.open
        with opener(scene_file, 'rb') as f:
            data = pickle.load(f)
            obs_pos = data['obs_pos']
            obs_size = data['obs_size']
            del data
        
        # If waypoints are provided, load them
        waypoints = None
        if self.override_waypoint_csv is not None:
            waypoint_file = os.path.join(exp_datapath, self.override_waypoint_csv)
            waypoints = self._load_waypoint_csv(waypoint_csv_path=waypoint_file)

        # get the stratified samples as a dict
        samples, strata_dists, strata = self._sample_strata(obs_pos, obs_size, waypoints=waypoints)
        centers, radii = self.sphere_fk.generate_select_spheres(samples)
        gt_spheres, gt_configs, gt_radii = self._compute_groundtruth_spheres_configs(centers, radii, obs_pos, obs_size)
        combined_gt_dict = {}
        for k in gt_spheres:
            combined_gt_dict[k] = {
                'config_sphere_collisions': gt_spheres[k],
                'config_collisions': gt_configs[k],
                'effective_radii': gt_radii[k]
            }

        self.state = {
            'joint_samples': samples,
            'strata_category': strata,
            'strata_dists': strata_dists,
            'strata_method': self.sample_stratification['method'],
            'centers': centers.cpu().numpy(),
            'radii': radii.cpu().numpy(),
            'gt_data': combined_gt_dict
        }

        # Save the state if a save path is provided
        if save_file is not None:
            opener = open if not self.compress else lzma.open
            save_file += '.xz' if self.compress else ''
            with opener(save_file, 'wb') as f:
                pickle.dump(self.state, f)
        
        # Dump a human readable table (TODO)
    
    def get_gt_spheres(self, gt_name):
        sphere_collisions = self.state['gt_data'][gt_name]['config_sphere_collisions']
        collisions = self.state['gt_data'][gt_name]['config_collisions']
        centers = self.state['centers']
        radii = self.state['gt_data'][gt_name]['effective_radii']
        # spheres are (n_configs, n_spheres, ...)
        return (centers, radii), sphere_collisions, collisions
    
    def _process_joint_sampling(self, robot: zpr.ZonoArmRobot, joint_sampling):
        from omegaconf import OmegaConf
        # Make sure joint sampling is a dictionary
        if OmegaConf.is_config(joint_sampling):
            joint_sampling = OmegaConf.to_object(joint_sampling)

        # Validate num and range, and convert to lists if dicts.
        all_joint_names = robot.urdf.actuated_joint_names
        if 'num' not in joint_sampling or joint_sampling['num'] is None:
            joint_sampling['num'] = 1
        if isinstance(joint_sampling['num'], dict):
            if set(joint_sampling['num'].keys()) != set(all_joint_names):
                raise ValueError("Joint sampling num must be specified for all joints using names.")
            joint_sampling['num'] = [joint_sampling['num'][joint] for joint in all_joint_names]
        elif isinstance(joint_sampling['num'], list):
            if len(joint_sampling['num']) != len(all_joint_names):
                raise ValueError("Joint sampling num must be specified for all joints if using a list.")
        if 'range' not in joint_sampling or joint_sampling['range'] is None:
            joint_sampling['range'] = robot.np.pos_lim.T
            joint_sampling['range'][robot.np.continuous_joints] = [-np.pi, np.pi]
            joint_sampling['range'] = joint_sampling['range'].tolist()
        if isinstance(joint_sampling['range'], dict):
            valid_keys = [k for k in joint_sampling['range'].keys() if joint_sampling['range'][k] is not None]
            diff = set(valid_keys) - set(all_joint_names)
            if diff:
                raise ValueError(f"Joint sampling range specified for joints not in the robot. Extra: {diff}")
            diff = set(all_joint_names) - set(joint_sampling['range'].keys())
            if diff:
                print(f"Joint sampling range not specified for all joints. Missing: {diff}. Using default range.")
                for joint in diff:
                    joint_idx = robot.urdf.actuated_joint_names.index(joint)
                    if robot.continuous_joints[joint_idx]:
                        joint_sampling['range'][joint] = [-np.pi, np.pi]
                    else:
                        joint_sampling['range'][joint] = robot.np.pos_lim[joint_idx].tolist()
            joint_sampling['range'] = [joint_sampling['range'][joint] for joint in all_joint_names]
        elif isinstance(joint_sampling['range'], list):
            if len(joint_sampling['range']) != len(all_joint_names):
                raise ValueError("Joint sampling range must be specified for all joints if using a list.")
        
        # if num and range are both single values, convert num to a list
        if isinstance(joint_sampling['num'], int) and isinstance(joint_sampling['range'], int):
            joint_sampling['num'] = [joint_sampling['num']] * len(all_joint_names)
        
        return sampling_config_generator(joint_sampling)
    
    def _validate_stratification(self, sample_stratification):
        from omegaconf import OmegaConf
        if OmegaConf.is_config(sample_stratification):
            sample_stratification = OmegaConf.to_object(sample_stratification)
        if 'method' not in sample_stratification:
            raise ValueError("Sample stratification must specify a method.")
        if sample_stratification['method'] not in ['center_dist', 'surface_dist']:
            raise ValueError("Invalid sample stratification method specified. Must be 'center_dist' or 'surface_dist'.")
        if sample_stratification['unit'] not in ['size', None]:
            raise ValueError("Invalid sample stratification unit specified. Must be 'size' for a multiple of obs_size or None for workspace units.")
        if 'num_per_strata' not in sample_stratification:
            raise ValueError("Sample stratification must specify num_per_strata (number of samples per strata).")
        if 'strata' not in sample_stratification or len(sample_stratification['strata']) == 0:
            raise ValueError("Sample stratification must specify strata.")
        return sample_stratification
        
    def _load_waypoint_csv(self, waypoint_csv_path):
        df = pd.read_csv(waypoint_csv_path)
        df['joint_angles'] = df['joint_angles'].apply(lambda x: np.fromstring(''.join(filter(lambda y: not y in '[]', x)), sep=',', dtype=float))
        return np.stack(df['joint_angles'].values)
        
    def _compute_groundtruth_spheres_configs(self, centers, radii, obs_pos, obs_size):
        groundtruth_labels_spheres = {}
        groundtruth_labels_radii = {}
        from collision_comparisons.geometry import box_sdf
        def min_sphere_dist(radii):
            dists = box_sdf(centers, obs_pos, obs_size)
            min_dists = torch.min(dists, dim=-1).values - radii
            return min_dists
        for k, v in self.computed_sets.items():
            if v is None:
                groundtruth_labels_spheres[k] = min_sphere_dist(radii)<=0
                groundtruth_labels_radii[k] = radii.cpu().numpy()
            elif isinstance(v, str):
                if v == "max":
                    radii_ = torch.max(radii, dim=-1, keepdim=True).values
                    groundtruth_labels_spheres[k] = min_sphere_dist(radii_)<=0
                    groundtruth_labels_radii[k] = radii_.cpu().numpy()
                elif v == "allmax":
                    radii_ = torch.max(radii)
                    groundtruth_labels_spheres[k] = min_sphere_dist(radii_)<=0
                    groundtruth_labels_radii[k] = radii_.cpu().numpy()
                else:
                    raise ValueError("Unknown computatation type for groundtruth labels " + v)
            elif isinstance(v, float):
                groundtruth_labels_spheres[k] = min_sphere_dist(v)<=0
                groundtruth_labels_radii[k] = v
            else:
                raise ValueError("Unknown computation type for groundtruth labels " + v)
        groundtruth_labels_spheres = {k: v.cpu().numpy() for k,v in groundtruth_labels_spheres.items()}
        groundtruth_labels_configs = {k: np.any(v, axis=-1) for k,v in groundtruth_labels_spheres.items()}
        return groundtruth_labels_spheres, groundtruth_labels_configs, groundtruth_labels_radii
        
    def _sample_strata(self, obs_pos, obs_size, waypoints=None):
        # test 1000 samples at a time
        from collision_comparisons.geometry import box_sdf, points_sdf

        # Create the bounds
        if self.sample_stratification['unit'] == 'size':
            size_unit = np.max(obs_size)
            ub = {k: size_unit*v if v is not None else np.inf for k,v in self.sample_stratification['strata'].items()}
        else:
            ub = {k: v if v is not None else np.inf for k,v in self.sample_stratification['strata'].items()}
        all_ub = np.array([-np.inf] + list(ub.values()))
        lb = {k: np.max(np.where(all_ub<v, all_ub, -np.inf)) for k,v in ub.items()}
        bounds = {k: (lb[k], ub[k]) for k in ub}

        # perform sampling until max samples or all strata are filled
        samples = {k: set() for k in self.sample_stratification['strata'].keys()}
        sampled = 0
        while sampled < self.max_sample_attempts:
            # do a single loop if waypoints are provided
            if waypoints is not None:
                joint_samples = waypoints
            else:
                joint_samples = self.joint_sampling(self.sample_size)
                sampled += self.sample_size

            joint_samples = torch.as_tensor(joint_samples, dtype=self.sphere_fk.dtype, device=self.sphere_fk.device)
            centers, radii = self.sphere_fk.generate_select_spheres(joint_samples)

            # center to center distance
            if self.sample_stratification['method'] == 'center_dist':
                dists = points_sdf(centers, obs_pos)
                min_dists = torch.min(dists, dim=-1).values.min(dim=-1).values
                for k, (l, u) in bounds.items():
                    if len(samples[k]) >= self.sample_stratification['num_per_strata']:
                        continue
                    mask = (min_dists >= l) & (min_dists < u)
                    add_sample_set = [(sample, dist) for sample, dist in zip(joint_samples[mask], min_dists[mask])]
                    samples[k] |= set(add_sample_set)
            
            # surface to surface distance
            elif self.sample_stratification['method'] == 'surface_dist':
                dists = box_sdf(centers, obs_pos, obs_size)
                min_dists = torch.min(dists, dim=-1).values - radii
                min_dists = torch.min(min_dists, dim=-1).values
                for k, (l, u) in bounds.items():
                    if len(samples[k]) >= self.sample_stratification['num_per_strata']:
                        continue
                    mask = (min_dists >= l) & (min_dists < u)
                    add_sample_set = [(sample, dist) for sample, dist in zip(joint_samples[mask], min_dists[mask])]
                    samples[k] |= set(add_sample_set)
            
            # check if all strata are filled with at least num_per_strata samples
            if all([len(v) >= self.sample_stratification['num_per_strata'] for v in samples.values()]):
                break
            
            # break out if waypoints are provided (single loop)
            if waypoints is not None:
                break
        # Make sure we only return the number of samples specified unless waypoints are provided
        if waypoints is not None:
            stratified_samples = {k: list(v) for k,v in samples.items()}
        else:
            stratified_samples = {k: list(v)[:self.sample_stratification['num_per_strata']] for k,v in samples.items()}

        # combine the samples and create the groundtruth bins
        samples = []
        strata_dists = []
        strata = []
        for k,v in stratified_samples.items():
            samples.extend([s.cpu().numpy() for s,_ in v])
            strata_dists.extend([d.cpu().numpy() for _,d in v])
            strata.extend([k]*len(v))
        samples = np.stack(samples)
        strata_dists = np.stack(strata_dists)
        return samples, strata_dists, strata

import importlib
import zonopyrobots as zpr
from omegaconf import OmegaConf
from planning.splanning.splanner import Splanner
from environments.urdf_obstacle import KinematicUrdfWithObstacles
from planning.armtd.armtd_3d_urdf import ARMTD_3D_planner
from planning.sparrows.sparrows_urdf import SPARROWS_3D_planner
from urchin import URDF
import torch
import numpy as np

from .config import CommonConfig

def create_robot(robot_config, common_config: CommonConfig):
    kwargs = {
        'device': common_config.device,
        'dtype': common_config.dtype
    }
    if "urdf" in robot_config.get("params", dict()):
        urdf = robot_config['params'].pop("urdf")
    if "urdf_ref" in robot_config and robot_config['urdf_ref']:
        urdf = _get_obj_from_str(robot_config['urdf_ref'])
    return instantiate_from_config(robot_config, args=(urdf,), kwargs=kwargs)


def setup_environment(
        env_filepath: str,
        robot: URDF | None = None,
        max_steps: int = 100,
        persistant_env: KinematicUrdfWithObstacles | None = None,
        check_self_collision: bool = False,
        visualize: bool = False,
    ):
    ''' Load the environment from a file and set it up with the robot provided.
    If a persistant environment is provided, it will be used if the robot matches the robot provided.
    If the persistant environment is set to visualize and the visualize flag is not set, the environment will be reinitialized.
    
    Args:
        env_filepath (str): The path to the environment file
        robot (URDF | None): The robot URDF to use for the environment
        persistant_env (KinematicUrdfWithObstacles | None): The persistant environment to use if the robot matches
        check_self_collision (bool): Whether to check self collision in the environment
        visualize (bool): Whether to visualize the environment
    
    Returns:
        KinematicUrdfWithObstacles: The environment
    '''
    import pickle, lzma

    opener = open
    if env_filepath.endswith('.xz'):
        opener = lzma.open
    
    with opener(env_filepath, 'rb') as f:
        data = pickle.load(f)
    
    env = None
    if persistant_env is not None:
        if persistant_env.robot != robot:
            print("Warning: Robot in persistant environment does not match the robot provided. Reinitializing environment.")
        if (persistant_env.renderer == 'pyrender-offscreen' and visualize) or (persistant_env.renderer == 'pyrender' and not visualize):
            print("Warning: Renderer in persistant environment does not match the visualization setting. Reinitializing environment.")
        else:
            env = persistant_env
            env.n_obs = data['n_obs_gen']
            env.check_self_collision = check_self_collision
            env._max_episode_steps = max_steps
            # These two can be set after the fact since n_obs is used for reset which we are doing
            # and regardless if check_self_collision is set or not, the self_collision manager is set up

    if robot is not None and env is None:
        env_common_args = dict(
                step_type='integration',
                check_joint_limits=True,
                use_bb_collision=False,
                render_mesh=True,
                reopen_on_close=False,
                obs_size_min = [0.2,0.2,0.2],
                obs_size_max = [0.2,0.2,0.2],
                info_nearest_obstacle_dist = False,
                obs_gen_buffer = 0.01,
            )
        env = KinematicUrdfWithObstacles(
                robot=robot,
                n_obs=data['n_obs_gen'],
                check_self_collision=check_self_collision,
                max_episode_steps=max_steps,
                renderer = 'pyrender-offscreen' if not visualize else 'pyrender',
                **env_common_args
            )
    
    if env is None:
        raise ValueError("No environment was created or provided. Was the robot provided?")
    
    # Reset the environment with the data
    env.reset(
        qpos=data['qpos'],
        qvel=data['qvel'],
        qgoal=data['qgoal'],
        obs_pos=data['obs_pos'],
        obs_size=data['obs_size'],
    )

    return env


def setup_splanner(
        robot: zpr.ZonoArmRobot,
        constraint_alpha: float = 0.1,
        constraint_beta: float = 0.1,
        device: torch.device = None,
        dtype: torch.dtype = None,
        joint_radius_override: dict[str, torch.Tensor] = {},
        # constraint_links: list[str] = [], # future feature
        spheres_per_link: int = 5,
        check_self_collisions: bool = False,
        linear_solver: str = 'ma57',
        planner_kd: float = 0.1,
        filter_links: bool = True,
    ):

    # Validate device and dtype
    dtype_device = torch.empty(0, device=device, dtype=dtype)
    device = dtype_device.device
    dtype = dtype_device.dtype
    info = {
        'name': 'splanner',
        'robot': robot.urdf.name,
        'device': str(device),
        'dtype': str(dtype),
    }
    
    planner_args = dict(
        use_weighted_cost=False,
        joint_radius_override=joint_radius_override,
        spheres_per_link=spheres_per_link,
        filter_links=filter_links,
        check_self_collisions=check_self_collisions,
        constraint_alpha=constraint_alpha,
        constraint_beta=constraint_beta,
        planner_kd=planner_kd,
        linear_solver=linear_solver,
    )
    info['planner_args'] = planner_args
    planner = Splanner(
        robot=robot,
        device=device,
        sphere_device=device,
        dtype=dtype,
        **planner_args
    )

    return planner, info


def splanner_setsplat(
        splanner: Splanner,
        splat_path: str,
        radial_culling_radius: float | None = None,
        radial_culling_sigma_mag: float | None = None,
        sigma_magnification: float = None,
    ):
    from planning.splanning.splat_tools import SplatLoader
    from planning.common.waypoints import ArmWaypoint
    splats = SplatLoader(
        filename=splat_path,
        device=splanner.sphere_device,
        dtype=splanner.dtype,
        # normalization_info=None,
    )
    if radial_culling_radius is not None:
        kwargs = {}
        if radial_culling_sigma_mag is not None:
            kwargs['cutoff_sigma_mag'] = radial_culling_sigma_mag
        splats.radial_culling([0,0,0], radial_culling_radius, copy=False, **kwargs)
    if sigma_magnification is not None:
        splats.precompute_errors_for_sigma_mag(sigma_magnification)
    splanner.set_splats(splats)

    # print("===Warming up Splanner constraint===")
    start = np.zeros(splanner.dof)
    goal = np.ones(splanner.dof)
    waypoint = ArmWaypoint(goal, start)
    splanner.plan(start, start, waypoint, None, time_limit=1)
    # print("===Done warming up Splanner constraint===")

    return {
        'path': splat_path,
        'radial_culling_radius': radial_culling_radius,
        'radial_culling_sigma_mag': radial_culling_sigma_mag,
        'sigma_magnification': splats.sigma_magnification,
    }


def setup_sparrows(
        robot: zpr.ZonoArmRobot,
        device: torch.device = None,
        dtype: torch.dtype = None,
        joint_radius_override: dict[str, torch.Tensor] = {},
        # constraint_links: list[str] = [], # future feature
        spheres_per_link: int = 5,
        check_self_collisions: bool = False,
        linear_solver: str = 'ma27',
        filter_links: bool = True,
    ):
    # Validate device and dtype
    dtype_device = torch.empty(0, device=device, dtype=dtype)
    device = dtype_device.device
    dtype = dtype_device.dtype
    info = {
        'name': 'sparrows',
        'robot': robot.urdf.name,
        'device': str(device),
        'dtype': str(dtype),
    }

    planner_args = dict(
        use_weighted_cost=False,
        joint_radius_override=joint_radius_override,
        spheres_per_link=spheres_per_link,
        filter_links=filter_links,
        check_self_collisions=check_self_collisions,
        linear_solver=linear_solver,
    )
    info['planner_args'] = planner_args
    planner = SPARROWS_3D_planner(
        robot,
        dtype=dtype,
        device=device,
        sphere_device=device,
        **planner_args
    )

    return planner, info


def setup_armtd(
        robot: zpr.ZonoArmRobot,
        device: torch.device = None,
        dtype: torch.dtype = None,
        filter_obstacles: bool = True,
        linear_solver: str = 'ma27',
    ):
    # Validate device and dtype
    dtype_device = torch.empty(0, device=device, dtype=dtype)
    device = dtype_device.device
    dtype = dtype_device.dtype
    info = {
        'name': 'armtd',
        'robot': robot.urdf.name,
        'device': str(device),
        'dtype': str(dtype),
    }

    planner_args = dict(
        filter_obstacles=filter_obstacles,
        linear_solver=linear_solver,
    )
    info['planner_args'] = planner_args
    planner = ARMTD_3D_planner(
        robot,
        device=device,
        **planner_args
    )

    return planner, info


def instantiate_from_config(config, args=(), kwargs={}):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    if OmegaConf.is_config(config):
        config = OmegaConf.to_object(config)
    return _get_obj_from_str(config["target"])(*args, **config.get("params", dict()), **kwargs)


def _get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    try:
        return getattr(importlib.import_module(module, package=None), cls)
    except ModuleNotFoundError as e:
        if '.' not in module:
            raise e
        return getattr(_get_obj_from_str(module, reload=reload), cls)


def sampling_config_generator(config):
    '''Generate a sampler function from a config.

    Supported methods:
        - uniform: Generate samples uniformly across a range
        - random: Generate random samples from a range
        - list: Generate samples from a list
        - log: Generate samples logarithmically across a range

    The config should have the following keys:
        - method: The method to use
        - range: The range to sample from as a list of two values. Can be a single
            range, a list of ranges, or a dictionary of ranges. If a list or dictionary,
            the number of values must match the number of values in num. Each range
            should be a list of two values. If the method is list, then each range
            entry should be a list of values to sample from.
    
    And optionally:
        - num (uniform, log, random): The number of discretizations for the range provided
            for uniform and logspace sampling. Can be a single value, a list of values,
            or a dictionary of values. If a list or dictionary, the number of values must
            match the number of values in the range. If a single value, the same number
            of samples will be generated for each range. Will be ignored for list.
            For random, this is the number of samples to generate.
        - base (log): The base for the logspace
        - all (log, uniform, list): If true, all permutations of the ranges will be
            generated and returned instead of a function that generates samples.

    Args:
        config (dict): The configuration for the sampler

    Returns:
        function: A function that generates samples
    '''
    # Avoid annoying DictConfig and ListConfig issues
    if OmegaConf.is_config(config):
        config = OmegaConf.to_object(config)

    # make sure expected keys are present
    if 'method' not in config:
        raise KeyError("Expected key `method` in sampling config.")
    if 'range' not in config:
        raise KeyError("Expected key `range` in sampling config.")
    if config['method'] in ['uniform', 'log'] and 'num' not in config:
        raise KeyError("Expected key `num` for uniform and log sampling.")
    if config['method'] == 'log' and 'base' not in config:
        raise KeyError("Expected key `base` for log sampling")
    if config['method'] in ['uniform', 'log', 'list'] and 'all' not in config:
        raise KeyError("Expected key `all` for uniform, log, and list sampling.")
    
    method = config['method']
    range_ = config['range']
    
    # Setup nums and ranges and validate if present
    if 'num' in config:
        num = config['num']

        keys_num = None
        keys_range = None
        if isinstance(num, dict):
            keys_num = list(num.keys())
        if isinstance(range_, dict):
            keys_range = list(range_.keys())
        if isinstance(num, list) and len(num) > 1:
            keys_num = len(num)
        if isinstance(range_, list) and len(range_) > 1 and isinstance(range_[0], list):
            keys_range = len(range_)
        single_num = keys_num is None
        single_range = keys_range is None
        equivalent = single_num == single_range
        if not (single_num or single_range or equivalent):
            raise ValueError("`num` and `range` must have equivalent keys, equivalent lists, or at least one of them must be a single value.")

        # expand num or range_ to match keys if necessary
        keys = keys_num if keys_num is not None else keys_range
        if single_num and not single_range:
            if isinstance(keys_range, int):
                num = [num] * keys_range
            else:    
                num = {k: num for k in keys_range}
        if single_range and not single_num:
            if isinstance(keys_num, int):
                range_ = [range_] * keys_num
            else:
                range_ = {k: range_ for k in keys_num}
    
    # otherwise keys is only informed by range_
    else:
        keys = None
        num = 1
        if isinstance(range_, list) and len(range_) > 1 and isinstance(range_[0], list):
            keys = len(range_)
            num = [1] * keys
        elif isinstance(range_, dict):
            keys = list(range_.keys())
            num = {k: 1 for k in keys}

    # expand all to lists for processing
    if isinstance(range_, dict):
        range_ = [range_[k] for k in range_]
    if isinstance(num, dict):
        num = [num[k] for k in num]
    
    if isinstance(num, int):
        num = [num]
        range_ = [range_]

    # helper function for permuting for uniform and logspace
    def permutator(func, num, range_):
        out = [func(n, r) for n, r in zip(num, range_)]
        if keys is None:
            return out[0]
        # Generate all permutations
        return np.stack(np.meshgrid(*out), axis=-1).reshape(-1, len(out))
    
    # helper function for list sampling (for sampling the precomputed permutations)
    def list_sampler(list_, n, replace=False):
        idxs = np.random.choice(len(list_), n, replace=replace).squeeze()
        rets = np.array(list_)[idxs]
        # map to keys if necessary
        if isinstance(keys, list):
            return {k: v for k, v in zip(keys, rets.T)}
        return rets
    
    # helper function to make sure we return the right format
    def return_transform(ret):
        if isinstance(keys, list):
            return {k: v for k, v in zip(keys, ret.T)}
        return ret

    ############################################
    # yield the uniform spread across the range
    if method == 'uniform':
        func = lambda n, r: np.linspace(r[0], r[1], n, endpoint=True)
        res = permutator(func, num, range_)
        if config['all']:
            return return_transform(res)
        else:
            return lambda n=1, replace=False: list_sampler(res, n, replace=replace)
    
    # yield samples from a logspace
    if method == 'log':
        func = lambda n, r: np.logspace(r[0], r[1], n, base=config['base'])
        res = permutator(func, num, range_)
        if config['all']:
            return return_transform(res)
        else:
            return lambda n=1, replace=False: list_sampler(res, n, replace=replace)
        
    # yield samples from a list
    if method == 'list':
        func = lambda n, r: np.array(r)
        res = permutator(func, num, range_)
        if config['all']:
            return return_transform(res)
        else:
            return lambda n=1, replace=False: list_sampler(res, n, replace=replace)
    
    # yield random samples from the range
    if method == 'random':
        r = np.array(range_).T
        return lambda n=num: return_transform(np.random.uniform(r[0], r[1], (n, len(r[0]))))
    
    

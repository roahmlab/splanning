from omegaconf import OmegaConf
from collections import namedtuple
from util.common import get_default_with_warning, validate_get_dtype_device
import os


DEFAULT_BASE_CONFIG = 'configs/base.yml'
'''The default base configuration file path.'''

def process_config(base_config: str = None):
    '''Process the configuration file and return the configuration tuple.
    
    The configuration is loaded from the base configuration file, then overlayed with the overlay configuration file,
    and finally the command line arguments are added on top. If the rerun flag is specified, then the rerun configuration
    file is used as the base configuration, and the overlay configuration file is ignored.
    
    Returns:
        ConfigTuple: The configuration tuple, where the fields are the full configuration, common configuration, robot configuration,
            planner configuration, splats configuration, and experiment configuration.
    '''
    cli_conf = OmegaConf.from_cli()
    # Base is loaded, then overlay, then cli is added on top
    # If rerun is specified, then rerun is treated as base, overlay is errored, and cli is added on top
    if base_config is None:
        base_conf_path = cli_conf.pop('base_config', DEFAULT_BASE_CONFIG)
    else:
        base_conf_path = base_config
        
    overlay_path = cli_conf.pop('overlay', None)
    rerun_path = cli_conf.pop('config', None)
    is_rerun = rerun_path is not None
    if is_rerun and overlay_path is not None:
        raise ValueError("Cannot specify both a rerun config and an overlay config.")
    if is_rerun:
        base_conf_path = rerun_path

    # Merge the configs
    config = OmegaConf.load(base_conf_path)
    if overlay_path is not None:
        overlay_config = OmegaConf.load(overlay_path)
        config = OmegaConf.merge(config, overlay_config)
    config = OmegaConf.merge(config, cli_conf)

    # Extract the parameters
    robot = get_default_with_warning(config, 'robot', dict())
    experiment = get_default_with_warning(config, 'experiment', dict())
    method = get_default_with_warning(experiment, 'method', None)
    if method is None:
        raise ValueError("Method must be specified in the experiment configuration.")
    planner = get_default_with_warning(config, method, dict())
    if method == 'splanning':
        splats = get_default_with_warning(config, 'splats', dict())
    else:
        splats = None

    # Validate
    if config.visualize and config.video:
        raise ValueError("Cannot save video and visualize at the same time.")
    if config.reachset_viz and not (config.visualize or config.video):
        print("Warning: Reachset visualization is enabled, but visualization is not enabled. Reachset visualization will be disabled.")
        config.reachset_viz = False
    

    ## setup common
    dtype, device = validate_get_dtype_device(
        get_default_with_warning(config, 'dtype', None),
        get_default_with_warning(config, 'device', None),
    )
    basepath = get_default_with_warning(config, 'basepath', None)
    if basepath is None or basepath == '':
        basepath = os.getcwd()
    config['basepath'] = basepath
    common = CommonConfig(**{
        'dtype': dtype,
        'device': device,
        'visualize': get_default_with_warning(config, 'visualize', False),
        'reachset_viz': get_default_with_warning(config, 'reachset_viz', False),
        'video': get_default_with_warning(config, 'video', False),
        'verbose': get_default_with_warning(config, 'verbose', False),
        'basepath': os.path.abspath(basepath),
    })

    return ConfigTuple(
        config,
        method,
        common,
        robot,
        planner,
        splats,
        experiment,
    )

ConfigTuple = namedtuple(
    'ConfigTuple', [
        'fullconfig',
        'method',
        'common',
        'robot',
        'planner',
        'splats',
        'experiment',
    ])
'''The configuration tuple, where the fields are the full configuration, common configuration, robot configuration,
    planner configuration, splats configuration, and experiment configuration.
    
    Attributes:
        fullconfig (dict): The full configuration dictionary.
        method (str): The method to use for the experiment.
        common (CommonConfig): The common configuration.
        robot (dict): The robot configuration.
        planner (dict): The planner configuration.
        splats (dict): The splats configuration.
        experiment (dict): The experiment configuration.
'''

CommonConfig = namedtuple(
    'CommonConfig', [
        'dtype',
        'device',
        'visualize',
        'reachset_viz',
        'video',
        'verbose',
        'basepath',
    ])
'''The common configuration tuple, where the fields are the data type, device, visualization flag, video flag, verbose flag, and base path.

    Attributes:
        dtype (torch.dtype): The data type for the tensors.
        device (torch.device): The device for the tensors.
        visualize (bool): Whether to visualize the environment.
        reachset_viz (bool): Whether to visualize the reachable set.
        video (bool): Whether to save a video of the visualization.
        verbose (bool): Whether to print verbose output.
        basepath (str): The base path for the scene, model, and output paths.
'''

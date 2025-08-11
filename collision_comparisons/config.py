import os
from omegaconf import OmegaConf
from util.common import get_default_with_warning, validate_get_dtype_device
from collections import namedtuple

DEFAULT_BASE_CONFIG = 'configs/collisions_baselines/base.yml'
def process_collision_config():
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
    default_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, DEFAULT_BASE_CONFIG)
    base_conf_path = cli_conf.pop('base_config', default_config_path)
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

    # Extract the configs
    robot = get_default_with_warning(config, 'robot', dict())
    sample_generation = get_default_with_warning(config, 'config_samples', dict())
    ground_truth = config.get('base_ground_truth', dict())
    splanning_config = get_default_with_warning(config, 'splanning', dict())
    splat_nav_config = get_default_with_warning(config, 'splat_nav', dict())
    catnips_config = get_default_with_warning(config, 'catnips', dict())

    # if override_names are specified for sample generation or ground truth, update accordingly
    if sample_generation.get('override_name', None) is not None:
        sample_generation = {'override_name': sample_generation["override_name"]}
        print(f'Sample generation overridden with results from run {sample_generation["override_name"]}')
    if ground_truth.get('override_name', None) is not None:
        ground_truth = {'override_name': ground_truth["override_name"]}
        print("Ground truth overridden, so sample generation will be skipped.")
        print(f'Ground truth overridden with results from run {ground_truth["override_name"]}')
        sample_generation = {}
    
    # Mark any method configs for skipping if skip is specified
    if splanning_config.get('skip', False):
        splanning_config = OmegaConf.create({'skip': True})
        print("Splanning constraint marked for skipping.")
    if splat_nav_config.get('skip', False):
        splat_nav_config = OmegaConf.create({'skip': True})
        print("Splat-Nav constraint marked for skipping.")
    if catnips_config.get('skip', False):
        catnips_config = OmegaConf.create({'skip': True})
        print("CATNIPS collisions marked for skipping.")

    ## setup common
    dtype, device = validate_get_dtype_device(
        get_default_with_warning(config, 'dtype', None),
        get_default_with_warning(config, 'device', None),
    )
    basepath = get_default_with_warning(config, 'basepath', None)
    if basepath is None or basepath == '':
        basepath = os.getcwd()
    config['basepath'] = basepath
    common = CommonConstraintConfig(**{
        'dtype': dtype,
        'device': device,
        'verbose': get_default_with_warning(config, 'verbose', False),
        'basepath': os.path.abspath(basepath),
        'scenes_path': get_default_with_warning(config, 'scenes_path', None),
        'selected_scenarios': get_default_with_warning(config, 'selected_scenarios', None),
        'datapath': get_default_with_warning(config, 'datapath', None),
        'seed': get_default_with_warning(config, 'seed', None),
        'prefix': get_default_with_warning(config, 'prefix', None),
    })

    return ConfigConstraintTuple(
        config,
        common,
        robot,
        sample_generation,
        ground_truth,
        splanning_config,
        splat_nav_config,
        catnips_config,
    )

ConfigConstraintTuple = namedtuple(
    'ConfigTuple', [
        'fullconfig',
        'common',
        'robot',
        'sample_generation',
        'ground_truth',
        'splanning',
        'splat_nav',
        'catnips',
    ])
'''The configuration tuple for constraint baseline comparisons.
    
    Attributes:
        fullconfig (dict): The full configuration dictionary.
        common (CommonConstraintConfig): The common configuration.
        robot (dict): The robot configuration.
        sample_generation (dict): The sample generation configuration.
        ground_truth (dict): The ground truth configuration.
        splanning (dict): The splanning constraint configuration.
        splat_nav (dict): The splat-nav constraint configuration.
        catnips (dict): The catnips constraint configuration.
'''

CommonConstraintConfig = namedtuple(
    'CommonConstraintConfig', [
        'dtype',
        'device',
        'verbose',
        'basepath',
        'scenes_path',
        'selected_scenarios',
        'datapath',
        'seed',
        'prefix',
    ])
'''The common configuration tuple for constraint baseline comparisons.

    Attributes:
        dtype (torch.dtype): The torch data type to use.
        device (torch.device): The torch device to use.
        verbose (bool): Whether to print verbose output.
        basepath (str): The base path for the experiment.
        scenes_path (str): The path to the scenes data.
        selected_scenarios (list): The list of scenarios to use.
        datapath (str): The path to the colllision data input/outputs.
        seed (int): The seed for the experiments.
        prefix (str): The prefix for the experiment.
'''

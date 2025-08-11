import torch
import numpy as np
import pandas as pd
from scipy import stats
import io
import os
from collections import namedtuple
from omegaconf import OmegaConf, ListConfig


ExperimentInfo = namedtuple(
    'ExperimentInfo', [
        'scenario_name',
        'relpath',
        'scene_file',
        'result_path',
    ])


class LogData:
    def __init__(self):
        self.result = 'INCOMPLETE'
        self.k = []
        self.q = []
        self.qd = []
        self.qdd = []
        self.flag = []
        self.trajectory = []
        self.planner_times = []
        self.planner_stats = {}
        self.collision_count = 0
        self.trajectory_dt = 0.01
        self.peak_gpu_memory = None
    
    @property
    def num_steps(self):
        return len(self.k)
    
    def trajectory_to_csv(self, path):
        all_qs = np.concatenate([self.trajectory[i]['q'] for i in range(self.num_steps)], axis=0)
        all_qds = np.concatenate([self.trajectory[i]['qd'] for i in range(self.num_steps)], axis=0)
        all_qdds = np.concatenate([self.trajectory[i]['qdd'] for i in range(self.num_steps)], axis=0)
        columns = ['t']
        for d in ['q', 'dq', 'ddq']:
            for i in range(all_qs.shape[1]):
                columns.append(f"{d}{i}")
        ts = np.arange(all_qs.shape[0]) * self.trajectory_dt
        df = pd.DataFrame(np.hstack((ts.reshape(-1, 1), all_qs, all_qds, all_qdds)), columns=columns)
        df.to_csv(path, index=False)
    
    def print_planning_stats(self, prefix=None, print_to_stdout=True, save_path=None):
        # appends to save_path if it exists otherwise creates a new file if not None
        output_buffer = io.StringIO()
        if prefix is not None:
            print(f'{prefix} Planning Stats:', self.result, file=output_buffer)
        else:
            print(f'Planning Stats:', self.result, file=output_buffer)
        print(f'Total time elasped with {self.num_steps} steps: {stats.describe(self.planner_times)}', file=output_buffer)
        print("Per step", file=output_buffer)
        for k, v in self.planner_stats.items():
            try:
                print(f'{k}: {stats.describe(v)}', file=output_buffer)
            except:
                pass
        print(f'Collided {self.collision_count} times', file=output_buffer)
        output = output_buffer.getvalue()
        output_buffer.close()
        if print_to_stdout:
            print(output)
        if save_path is not None:
            with open(save_path, 'a') as f:
                f.write(output)
        return output
    
    def convert_to_dict_list(self, ignore_keys=[]):
        dict = {
            k: v for k,v in self.__dict__.items()
            if not (
                k.startswith('_') or callable(v) or k in ignore_keys
            )}
        return convert_dict_to_dict_list(dict)


def build_experiment_list(basepath, scenes_path, results_path, selected_scenarios = None):
    '''Build a list of dictionaries containing information about the experiments to run by recursively searching for scene files.
    
    Args:
        basepath (str): The base path for the experiments
        scenes_path (str): The path to the scenarios
        results_path (str): The path to the results
        selected_scenarios (list[str] | None): The list of scenarios to run, or None to run all found
    
    Returns:
        list[dict]: A list of dictionaries containing the scenario name, relative path for hierchical storage, scene file path, and results path
    '''
    
    # helper to add the correct extension to a scene file
    def add_correct_ext(scene_file_no_ext):
        if os.path.isfile(scene_file_no_ext + '.pickle'):
            return scene_file_no_ext + '.pickle'
        elif os.path.isfile(scene_file_no_ext + '.pickle.xz'):
            return scene_file_no_ext + '.pickle.xz'
        elif os.path.isdir(scene_file_no_ext):
            # look for a data.pickle file
            intermediate = os.path.join(scene_file_no_ext, 'data.pickle')
            if os.path.isfile(intermediate):
                return intermediate
            elif os.path.isfile(intermediate + '.xz'):
                return intermediate + '.xz'
            return scene_file_no_ext
        else:
            raise FileNotFoundError(f"Scene file {scene_file_no_ext} not found.")
    
    # helper to strip the extension from a scene file
    def strip_ext(scene_file):
        if scene_file.endswith('data.pickle') or scene_file.endswith('data.pickle.xz'):
            return os.path.dirname(scene_file)
        elif scene_file.endswith('.pickle'):
            return scene_file[:-7]
        elif scene_file.endswith('.pickle.xz'):
            return scene_file[:-10]
        elif os.path.isdir(scene_file):
            return scene_file
        else:
            raise ValueError(f"Scene file {scene_file} has an unexpected extension.")

    # helper for search finding scene_files
    def walker_helper(search_path):
        scene_files = []
        full_search_path = os.path.join(basepath, search_path)
        for folder_path, _, filenames in os.walk(full_search_path):
            scene_files.extend([
                os.path.join(folder_path, f) for f in filenames
                if f.endswith('.pickle') or f.endswith('.pickle.xz') or f.endswith('data.pickle') or f.endswith('data.pickle.xz')
            ])
        return scene_files
    
    # if selected_scenarios has any folders instead of files, search for all files in those folders
    if selected_scenarios is None:
        scene_files = walker_helper(scenes_path)
    else:
        if not isinstance(selected_scenarios, (list, ListConfig)):
            selected_scenarios = [selected_scenarios]
        scene_files_unfiltered = [add_correct_ext(os.path.join(basepath, scenes_path, f)) for f in selected_scenarios]
        scene_files = []
        for name, file in zip(selected_scenarios, scene_files_unfiltered):
            if not os.path.isfile(file):
                scene_files_inside = walker_helper(os.path.join(scenes_path, name))
                scene_files.extend(scene_files_inside)
            else:
                scene_files.append(file)
    
    search_path = os.path.join(basepath, scenes_path)
    selected_scenarios = [strip_ext(os.path.relpath(f, search_path)) for f in scene_files]

    # remove possible duplicates from scene_files and selected_scenarios using selected_scenarios as the order
    scenes = {}
    for name, file in zip(selected_scenarios, scene_files):
        # prefer the pickle file over the compressed pickle file
        if name in scenes and scenes[name].endswith('.pickle'):
            continue
        scenes[name] = file
    scene_files = [scenes[name] for name in selected_scenarios]
    selected_scenarios = list(scenes.keys())

    results_paths = [os.path.join(basepath, results_path, relpath) for relpath in selected_scenarios]
    experiment_dict_list = [ExperimentInfo(**{
        'scenario_name': name,
        'relpath': name,
        'scene_file': file,
        'result_path': res_path,
    }) for name, file, res_path in zip(selected_scenarios, scene_files, results_paths)]
    experiment_dict_list.sort(key=lambda x: alphanum_key(x.scenario_name))
    
    return experiment_dict_list

### Natural sorting from Ned B
import re

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

###

def convert_dict_to_dict_list(in_dict):
    out_dict = {}
    def convert_helper(val):
        from omegaconf import DictConfig
        if isinstance(val, (np.ndarray, torch.Tensor)):
            return val.tolist()
        if isinstance(val, list):
            return [convert_helper(v) for v in val]
        if isinstance(val, (dict, DictConfig)):
            return convert_dict_to_dict_list(val)
        return val
    for key, val in in_dict.items():
        out_dict[key] = convert_helper(val)
    return out_dict


def get_splat_file(experiment_info, splats_config, basepath):
    '''Get the splat file for the given experiment information and configuration.
    
    Args:
        experiment_info (ExperimentInfo): The experiment information
        splats_config (dict): The configuration dictionary for the splats
    
    Returns:
        str: The path to the splat file to load
        dict: The configuration dictionary for the splat to load
    '''

    if OmegaConf.is_config(splats_config):
        splats_config = OmegaConf.to_object(splats_config)
    
    splats_folder = os.path.join(basepath, splats_config['models_path'], experiment_info.relpath)
    splats_folder = os.path.join(splats_folder, splats_config['splat_subpath'])

    splats_config = splats_config.copy()
    splats_config.pop('models_path')
    splats_config.pop('splat_subpath')

    return splats_folder, splats_config

def get_nerfstudio_config(experiment_info, nerfs_config, basepath):
    '''Get the splat file for the given experiment information and configuration.
    
    Args:
        experiment_info (ExperimentInfo): The experiment information
        nerfs_config (dict): The configuration dictionary for the nerf
    
    Returns:
        str: The path to the splat file to load
        dict: The configuration dictionary for the splat to load
    '''

    if OmegaConf.is_config(nerfs_config):
        nerfs_config = OmegaConf.to_object(nerfs_config)
    
    nerfstudio_folder = os.path.join(basepath, nerfs_config['models_path'])
    nerfstudio_folder = os.path.join(nerfstudio_folder, experiment_info.relpath, 'depth-nerfacto')
    
    # Stolen from stack overflow forever ago
    def sorted_nicely( l ): 
        """ Sort the given iterable in the way that humans expect.""" 
        convert = lambda text: int(text) if text.isdigit() else text 
        alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
        return sorted(l, key = alphanum_key)

    run_folder = sorted_nicely(os.listdir(nerfstudio_folder))[-1]
    
    nerf_config_file = os.path.join(nerfstudio_folder, run_folder)

    return nerf_config_file

    
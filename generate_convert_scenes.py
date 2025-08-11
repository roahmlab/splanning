''' This script is used to generate or convert scenes for each of the 3 different number of obstacles (10, 20, 40) for the
    Kinova Gen3 robot arm. The scenes are generated using the SPARROWS environment generation code. The scenes are then
    saved as pickle files in the 'scenes' folder. '''
import zonopyrobots as zpr
import numpy as np
import os

# Load robot URDF
robot = zpr.robots.urdfs.KinovaGen3

# Parameters for generating scenes based on SPARROWS
do_sparrows_scenarios = True
load_sparrows_scenario_path = None # Can be set to a path to load the SPARROWS scenarios from, otherwise they will be generated to match SPARROWS
n_obs_list = (10, 20, 40)
n_scenes_per_obs = 100
add_table = False

# Parameters for loading the handcrafted scenes from SPARROWS
do_hard_scenarios = False # we don't do these
hard_scenarios_folder = os.path.join(os.path.dirname(__file__), 'kinova_scenarios')

# Where to dump
output_folder = os.path.join(os.path.dirname(__file__), 'scenes')
compress_output = True
skip_if_exists = False # in case resume is needed instead
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


def generate_sparrows_scenarios(robot_urdf, n_envs = 100, n_obs = 10, check_self_collision = False):
    # SPARROWS generates environments randomly but using consistent seeds. To ensure that the environments are generated
    # consistently with how SPARROWS generated their environments, we recreate the the environment generation routine
    # and then export the desired values to recreate them. There is one key detail here, which is that the seed internal
    # to SPARROWS does not get reset properly, so we have to pregenerate the data
    from environments.urdf_obstacle import KinematicUrdfWithObstacles
    import torch
    import random
    def set_random_seed(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    env_args = dict(
            step_type='integration',
            check_joint_limits=True,
            check_self_collision=check_self_collision,
            use_bb_collision=False,
            render_mesh=True,
            reopen_on_close=False,
            obs_size_min = [0.2,0.2,0.2],
            obs_size_max = [0.2,0.2,0.2],
            n_obs=n_obs,
            renderer = 'pyrender-offscreen',
            info_nearest_obstacle_dist = False,
            obs_gen_buffer = 0.01
        )
    env = KinematicUrdfWithObstacles(
            robot=robot_urdf,
            **env_args
        )
    
    environment_data = []
    for i_env in range(n_envs):
        set_random_seed(i_env)
        obs = env.reset()
        environment_data.append({
            'qpos': obs['qpos'],
            'qvel': obs['qvel'],
            'qgoal': obs['qgoal'],
            'obs_pos': obs['obstacle_pos'],
            'obs_size': obs['obstacle_size'],
            'n_obs_gen': env.n_obs,
            'n_obs_final': len(obs['obstacle_pos']),
        })
    env.close()
    return environment_data


def load_sparrows_hard_scenarios(input_folder):
    import csv
    # Enumerate the files in the input folder and remove the .csv extension
    input_files = [name[:-4] for name in os.listdir(input_folder) if name.endswith('.csv')]
    environment_data = {}
    for scene_name in input_files:
        print(f'Loading scene {scene_name}')
        obs = []
        with open(os.path.join(input_folder, scene_name + '.csv'), mode ='r') as file:
            csvFile = csv.reader(file)
            line_number = 0
            for line in csvFile:
                if line_number == 0:
                    qstart = np.array([float(num) for num in line if num != 'NaN'])
                elif line_number == 1:
                    qgoal = np.array([float(num) for num in line])
                elif line_number > 2:
                    obs.append([float(num) for num in line][:3])
                line_number += 1
            n_obs = len(obs)
        environment_data[scene_name] = {
            'qpos': qstart,
            'qvel': qstart*0,
            'qgoal': qgoal,
            'obs_pos': obs,
            'obs_size': [[0.2, 0.2, 0.2] for _ in range(n_obs)],
            'n_obs_gen': n_obs,
            'n_obs_final': n_obs,
        }
    return environment_data


def load_sparrows_run_scenarios(input_folder, n_obs, n_scenes_per_obs):
    # Alternately, recreate the initial states from the SPARROWS runs
    import pickle
    # load the initial from the input folder
    filename = f'armtd_1branched_t0.5_stats_3d7links{n_scenes_per_obs}trials{n_obs}obs150steps_0.5limit.pkl'
    with open(os.path.join(input_folder, f'3d7links{n_obs}obs', filename), 'rb') as handle:
        data = pickle.load(handle)
    initials = [data[i]['initial'] for i in range(n_scenes_per_obs)]
    environment_data = [{
        'qpos': initial['qpos'],
        'qvel': initial['qvel'],
        'qgoal': initial['qgoal'],
        'obs_pos': initial['obstacle_pos'],
        'obs_size': initial['obstacle_size'],
        'n_obs_gen': len(initial['obstacle_pos']),
        'n_obs_final': len(initial['obstacle_pos']),
    } for initial in initials]
    return environment_data


def create_multiview_data(robot_urdf, env_config_dict, add_table=False, n_extra_cameras=20, render_floor=True, render_walls=True):
    '''Create multiview data from the environment configuration dictionary'''
    from environments.urdf_multiview import KinematicUrdfWithObstaclesMultiview
    env = KinematicUrdfWithObstaclesMultiview(
        robot = robot_urdf,
        step_type='integration',
        check_joint_limits=True,
        check_self_collision=True,
        use_bb_collision=True,
        render_mesh=True,
        reopen_on_close=False,
        obs_size_min = [0.2,0.2,0.2],
        obs_size_max = [0.2,0.2,0.2],
        render_fps=30,
        render_frames=1,
        n_extra_cams=n_extra_cameras,
        viz_goal=False,
        n_obs=env_config_dict['n_obs_gen'],
        add_render_floor=render_floor,
        add_render_walls=render_walls,
        # seed=seed,
        render_floor_height=-0.75,
        render_wall_dist=2.5,
        render_wall_height=5,
        renderer_kwargs={
            'relative_camera_pose':
                [[-1.2246467991473532e-16, 0.41, -0.91, -1.5596213788353745],
                [-1.0, -5.021051876504148e-17, 1.1144285872240914e-16, 0.0],
                [0.0, 0.91, 0.41, 0.5198737929451248],
                [0.0, 0.0, 0.0, 1.0]]
            }
        )
    
    # Setup
    qpos = env_config_dict['qpos']
    qvel = env_config_dict['qvel']
    qgoal = env_config_dict['qgoal']
    obs_pos = env_config_dict['obs_pos']
    obs_size = env_config_dict['obs_size']
    env.reset(qpos=qpos, qvel=qvel, qgoal=qgoal, obs_pos=obs_pos, obs_size=obs_size)
    if add_table:
        # Make it slightly lower than 0 just to make sure it's below the robot
        env.add_obstacle(np.array([0, 0, -0.251]), np.array([0.5, 0.5, 0.5]))

    # Generate data
    im, depth = env.render(hide_robot=True)
    data = {
        'rgb': im[0],
        'depth': depth[0],
        'poses': env.get_observations()['extra_cameras'],
        # Needed to consistently regen environment along with robot urdf
        'n_obs_final': len(env.get_observations()['obstacle_pos']),
        'obs_pos': env.get_observations()['obstacle_pos'],
        'obs_size': env.get_observations()['obstacle_size'],
        'qpos': qpos,
        'qvel': qvel,
        'qgoal': qgoal,
        # extras
        'n_obs_gen': env.n_obs,
        }
    env.close()
    return data


def test_preview(data):
    '''Preview the data'''
    import matplotlib.pyplot as plt
    plt.clf()
    f = plt.gcf()
    # plt.imshow(data['rgb'][0])
    # plt.show()
    axarr = f.subplots(4,4)
    axarr[0][0].imshow(data['rgb'][0])
    axarr[1][0].imshow(data['rgb'][1])
    axarr[2][0].imshow(data['rgb'][2])
    axarr[3][0].imshow(data['rgb'][3])
    axarr[0][1].imshow(data['rgb'][0+4])
    axarr[1][1].imshow(data['rgb'][1+4])
    axarr[2][1].imshow(data['rgb'][2+4])
    axarr[3][1].imshow(data['rgb'][3+4])
    axarr[0][2].imshow(data['rgb'][0+8])
    axarr[1][2].imshow(data['rgb'][1+8])
    axarr[2][2].imshow(data['rgb'][2+8])
    axarr[3][2].imshow(data['rgb'][3+8])
    axarr[0][3].imshow(data['rgb'][0+12])
    axarr[1][3].imshow(data['rgb'][1+12])
    axarr[2][3].imshow(data['rgb'][2+12])
    axarr[3][3].imshow(data['rgb'][3+12])
    plt.draw()
    plt.show()


def pickle_dumper(compress=False, filename="scene", dump_path=os.path.dirname(__file__), overwrite=False):
    '''Helper to dump data to pickle'''
    import pickle
    import lzma
    if not os.path.exists(dump_path):
        os.makedirs(dump_path)
    fullfilename = f'{filename}.pickle'
    if compress:
        fullfilename = fullfilename + '.xz'
        opener = lzma.open
    else:
        opener = open
    if not overwrite and os.path.exists(os.path.join(dump_path, fullfilename)):
        return None
    def dumper(data):
        with opener(os.path.join(dump_path, fullfilename), 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return fullfilename
    return dumper


if do_hard_scenarios:
    print("Loading hard scenarios")
    data_dict = load_sparrows_hard_scenarios(hard_scenarios_folder)
    for scene_name, data in data_dict.items():
        dumper = pickle_dumper(compress=compress_output, filename=scene_name, dump_path=os.path.join(output_folder, "hard"), overwrite=not skip_if_exists)
        if dumper is not None:
            data_out = create_multiview_data(robot, data)
            fileout = dumper(data_out)
            print(f"Saved {fileout}")
        else:
            print(f"Skipped {scene_name}. File exists.")

if do_sparrows_scenarios:
    print("SPARROWS scenarios")
    for n_obs in n_obs_list:
        if load_sparrows_scenario_path is not None:
            print("Loading for", n_obs, "obstacles")
            data_list = load_sparrows_run_scenarios(load_sparrows_scenario_path, n_obs, n_scenes_per_obs)
        else:
            print("Generating for", n_obs, "obstacles")
            data_list = generate_sparrows_scenarios(robot, n_envs=n_scenes_per_obs, n_obs=n_obs)
        for i, data in enumerate(data_list):
            dumper = pickle_dumper(compress=compress_output, filename=f'scene_{i}', dump_path=os.path.join(output_folder, f'{n_obs}obs'), overwrite=not skip_if_exists)
            if dumper is not None:
                data_out = create_multiview_data(robot, data)
                fileout = dumper(data_out)
                print(f"Saved {fileout}")
            else:
                print(f"Skipped scene_{i} for {n_obs} obstacles. File exists.")

print("Done!")

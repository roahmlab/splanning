import argparse

import numpy as np
import torch

import sys, os

import yaml
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from planning.splanning.splanner import T_FULL, T_PLAN, Splanner
from planning.splanning.splat_tools import SplatLoader
from planning.common.waypoints import ArmWaypoint

torch.set_float32_matmul_precision('high')

def parse_args():
    parser = argparse.ArgumentParser(description="Script Configuration")
    parser.add_argument('scene_dir', type=str, help="Path to experiment scene data")


    parser.add_argument('--config', type=str, help="Path to the configuration file (YAML format)")
    
    # Arguments for method
    parser.add_argument('--method', type=str, default="splanning", help="Method to use")

    # Arguments for scene
    parser.add_argument('--iteration_number', type=int, default=30000, help="Number of iterations")
    parser.add_argument('--log_dir', type=str, default=None, help="Directory to save results. Defaults to <project_root>/planning_results")

    parser.add_argument('--opt_wp_rad', type=int, default=5, help="Optimal waypoint radius")


    # Arguments for optimizer
    parser.add_argument('--num_steps', type=int, default=40, help="Number of optimization steps")
    parser.add_argument('--linear_solver', type=str, default='ma57', help="Linear solver to use")
    parser.add_argument('--planner_kd', type=float, default=0.1, help="KD value for optimizer")
    parser.add_argument('--time_limit', type=float, default=0.5, help="Time limit for each optimizer solve")

    # Arguments for constraint
    parser.add_argument('--constraint_alpha', type=float, default=0.05, help="Alpha for constraint")
    parser.add_argument('--constraint_beta', type=float, default=0.01, help="Beta for constraint")
    parser.add_argument('--sigma_region', type=int, default=6, help="Sigma region for constraint")


    parser.add_argument('--visualize', action="store_true", help="If set, render visualization")
    parser.add_argument('--debug', action="store_true", help="If set, render visualization")
    parser.add_argument('--self_collision', action="store_true", help="whether sim should check self-collisions.")

    args = parser.parse_args()
    
    if args.config:
        with open(args.config, 'r') as f:
            config_data = yaml.full_load(f)
        
        for k,v in config_data:
            setattr(args, k, v)

    return args

if __name__ == '__main__':
    args = parse_args()
    print(args)
    
    planning_method = args.method
    assert planning_method == "splanning"

    scene_path = args.scene_dir

    constraint_rhs = args.constraint_beta
    constraint_alpha = args.constraint_alpha
    
    linear_solver = args.linear_solver
    planner_kd = args.planner_kd
    num_steps = args.num_steps

    log_data = {
        "trajectory": {
            "q": [],
            "qd": [],
            "qdd": [],
            "k": []
        },
        "result": None,
        "experiment_data": args.scene_dir,
        "rhs": args.constraint_beta,
    }

    if args.log_dir is not None:
        log_dir = args.log_dir
    else:
        import datetime
        now = datetime.datetime.now()
        now_str = now.strftime("%m%d%y_%H%M%S")
        log_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'planning_results', now_str)

    os.makedirs(log_dir, exist_ok=True)

    from environments.urdf_obstacle import KinematicUrdfWithObstacles
    import time
    ##### 0.SET DEVICE #####
    if torch.cuda.is_available():
    # if False:
        device = 'cuda:0'
        #device = 'cpu'
        dtype = torch.float
        # dtype = torch.double
    else:
        device = 'cpu'
        dtype = torch.float
        # dtype = torch.double

    ##### LOAD ROBOT #####
    import os

    print('Loading Robot')
    # This is hardcoded for now
    import zonopyrobots as robots2
    basedirname = os.path.dirname(robots2.__file__)

    robots2.DEBUG_VIZ = False
    rob = robots2.ZonoArmRobot.load(os.path.join(basedirname, 'robots/assets/robots/kinova_arm/gen3.urdf'), device=device, dtype=dtype, create_joint_occupancy=True)

    ##### SET ENVIRONMENT #####
    env = KinematicUrdfWithObstacles(
        robot=rob.urdf,
        step_type='integration',
        check_joint_limits=True,
        check_self_collision=False,
        use_bb_collision=False,
        render_mesh=True,
        render_fps = 10,
        render_frames = 5,
        reopen_on_close=False,
        obs_gen_buffer = 1e-4,
        info_nearest_obstacle_dist = False,
        check_collision=True,
        renderer = 'pyrender' if args.visualize else 'pyrender-offscreen',
        )
    
    splat_path = f"{scene_path}/splats/csv/iteration_30000.csv"
    splats = SplatLoader(
        filename=splat_path,
        device=device,
        dtype=dtype,
        )
    splats.radial_culling([0,0,0], 0.902, copy=False)
    
    with open(f"{scene_path}/data.pickle", 'rb') as f:
        import pickle
        env_data = pickle.load(f)
        obs_pos = env_data['obs_pos']
        obs_size = env_data['obs_size']
        qpos = env_data['qpos']
        qgoal = env_data['qgoal']
        del env_data

    obs = env.reset(qpos=qpos, qgoal=qgoal, obs_pos = obs_pos, obs_size = obs_size)

    num_ks = num_steps + 1
    tqdm_steps = num_steps

    # env.add_obstacle(np.array([0, 0, -0.251]), np.array([0.5, 0.5, 0.5]))

    log_data.update({
        "start": obs['qpos'],
        "goal": obs["qgoal"]
    })

    from environments.fullstep_recorder import FullStepRecorder
    recorder = FullStepRecorder(env, path=os.path.join(os.getcwd(), log_dir, f"trajectory.mp4"))

    ##### 2. RUN ARMTD #####
    joint_radius_override = {
        'joint_1': torch.tensor(0.0503305, dtype=dtype, device=device),
        'joint_2': torch.tensor(0.0630855, dtype=dtype, device=device),
        'joint_3': torch.tensor(0.0463565, dtype=dtype, device=device),
        'joint_4': torch.tensor(0.0634475, dtype=dtype, device=device),
        'joint_5': torch.tensor(0.0352165, dtype=dtype, device=device),
        'joint_6': torch.tensor(0.0542545, dtype=dtype, device=device),
        'joint_7': torch.tensor(0.0364255, dtype=dtype, device=device),
        'end_effector': torch.tensor(0.0394685, dtype=dtype, device=device),
    }

    log_data["joint_radii"] = joint_radius_override

    # rob.np.vel_lim = np.array([0.43, 0.43, 0.43, 0.43, 4.3, 4.3, 4.3])
    planner = Splanner(rob, device=device, sphere_device=device, dtype=dtype, 
                                    use_weighted_cost=False, joint_radius_override=joint_radius_override, 
                                    spheres_per_link=5, filter_links=True, 
                                    check_self_collisions=args.self_collision,
                                    splats=splats,
                                    constraint_alpha = constraint_alpha, constraint_beta=constraint_rhs,
                                    planner_kd=planner_kd,
                                    linear_solver=linear_solver)

    start = np.zeros(7,)
    goal = np.ones(7,)
    waypoint = ArmWaypoint(goal, start)

    print("===Warming up planner===")
    planner.plan(start, start, waypoint, None, time_limit=1)
    print("===Done warming up planner===")

    from visualizations.sphere_viz import SpherePlannerViz
    plot_full_set = True
    sphereviz = SpherePlannerViz(planner, plot_full_set=plot_full_set, t_full=T_FULL)
    env.add_render_callback('spheres', sphereviz.render_callback, needs_time=not plot_full_set)

    use_last_k = False
    t_armtd = []
    T_NLP = []
    T_SFO = []
    T_NET_PREP = []
    T_PREPROC = []
    T_CONSTR_E = []
    N_EVALS = []

    #### Setup Waypoint Generator ####
    # Goal doesn't change so just set it here
    from planning.common.waypoints import GoalWaypointGenerator

    waypoint_generator = GoalWaypointGenerator(obs['qgoal'], planner.osc_rad*5)

    total_stats = {}
    count = 0
    ka = np.zeros(rob.dof)
    from tqdm import tqdm
    stuck = False

    last_step = False
    first_step = True
    
    for i in tqdm(range(num_ks), dynamic_ncols=True, total=tqdm_steps):
        ts = time.time()
        qpos, qvel = obs['qpos'], obs['qvel']
        obstacles = (np.asarray(obs['obstacle_pos']), np.asarray(obs['obstacle_size']))
        if i < num_steps:
            wp = waypoint_generator.get_waypoint(qpos, qvel, qgoal=obs['qgoal'])

            time_limit = None if first_step else args.time_limit
            first_step = False
            
            ka, flag, stats = planner.plan(qpos, qvel, wp, obstacles, 
                                        ka_0=(ka if use_last_k else None), 
                                        time_limit=time_limit,
                                        debug=args.debug)

            # print(planner.nlp_problem_obj._Cons[-1])
            if flag == 0:
                sphereviz.set_ka(ka)
                stuck = False
            else:
                sphereviz.set_ka(None)
                if stuck:
                    log_data["result"] = "STUCK"
                    break
                stuck = True

            t_elasped = time.time()-ts
            t_armtd.append(t_elasped)
            
            for k, v in stats.items():
                if v is None:
                    continue
                if k not in total_stats:
                    total_stats[k] = []
                if k == 'constraint_times':
                    total_stats[k] += v
                    N_EVALS.append(len(v))
                else:
                    total_stats[k].append(v)
            if flag != 0 and args.debug:
                print("executing failsafe")
        else:
            if args.debug:
                print("Executing brake at end of traj")
            flag = -1

        if flag != 0:
            ka = (0 - qvel)/(T_FULL - T_PLAN)
        log_data["trajectory"]['k'].append(ka)

        obs, rew, done, info, traj = env.step(ka, True)
        log_data["trajectory"]["q"].append(traj['q'])
        log_data["trajectory"]["qd"].append(traj['qd'])
        log_data["trajectory"]["qdd"].append(traj['qdd'][::2])

        # env.step(ka,flag)
        # assert(not info['collision_info']['in_collision'])
        # env.render()
        recorder.capture_frame()
        if 'collision_info' in info and info['collision_info']['in_collision']:
            print("collided!")
            log_data["result"] = "COLLISION"
            count += 1

            
        if rew and log_data["result"] != "COLLISION":
            log_data["result"] = "SUCCESS"
            break
        
    if args.visualize:
        env.spin(wait_for_enter=True)
    
    if log_data["result"] is None:
        log_data["result"] = "INCOMPLETE"

    all_qs = np.vstack(log_data["trajectory"]["q"])
    all_qds = np.vstack(log_data["trajectory"]["qd"])
    all_qdds = np.vstack(log_data["trajectory"]["qdd"])

    import pandas as pd
    columns = ['t']
    for d in ['q', 'dq', 'ddq']:
        for i in range(7):
            columns.append(f"{d}{i}")
    ts = np.arange(all_qs.shape[0]) / 100
    df = pd.DataFrame(np.hstack((ts.reshape(-1, 1), all_qs, all_qds, all_qdds)), columns=columns)
    df.to_csv(f"{log_dir}/traj.csv", index=False)

    log_data["run_stats"]         = total_stats
    log_data["num_collisions"]    = count
    log_data["trajectory"]['q']   = all_qs
    log_data["trajectory"]['qd']  = all_qds
    log_data["trajectory"]['qdd'] = all_qdds
    log_data["trajectory"]['k']   = np.vstack(log_data["trajectory"]['k'])

    from scipy import stats
    print("Planning Result:", log_data["result"])
    print(f'Total time elasped for ARMTD-3D with {num_steps} steps: {stats.describe(t_armtd)}')
    print("Per step")
    for k, v in total_stats.items():
        try:
            print(f'{k}: {stats.describe(v)}')
        except:
            pass
    if len(N_EVALS) > 0:
        print(f'number of constraint evals: {stats.describe(N_EVALS)}')
    print(f'Collided {count} times')
    
    
    recorder.close()

    def convert_ndarray_to_list(d):
        for key, value in d.items():
            if isinstance(value, np.ndarray):
                d[key] = value.tolist()
            elif isinstance(value, torch.Tensor):
                d[key] = value.tolist()
            elif isinstance(value, dict):
                convert_ndarray_to_list(value)
        return d

    import json
    with open(f"{log_dir}/data.json", 'w+') as f:
        json.dump(convert_ndarray_to_list(log_data), f, indent=2)

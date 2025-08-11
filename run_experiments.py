from omegaconf import OmegaConf
from util import (
    process_config,
    create_robot,
    setup_armtd,
    setup_sparrows,
    setup_splanner,
    splanner_setsplat,
    setup_environment,
    LogData,
    build_experiment_list,
    convert_dict_to_dict_list,
    get_splat_file,
)
import os
from visualizations.fo_viz import FOViz
from visualizations.sphere_viz import SpherePlannerViz
from tqdm import tqdm
from planning.common.waypoints import GoalWaypointGenerator
from environments.fullstep_recorder import FullStepRecorder
import time
import numpy as np
import json
import torch
T_PLAN, T_FULL = 0.5, 1.0


def main():
    # Process file configs and command line arguments
    code_dir = os.path.dirname(os.path.abspath(__file__))
    base_config_path = os.path.join(code_dir, 'configs', 'experiments.yml')
    configs = process_config(base_config_path)

    # helper function for printing
    def print_helper(*args, **kwargs):
        if configs.common.verbose:
            tqdm.write(*args, **kwargs)
    
    # Date and summary storage
    run_date = time.strftime("%Y%m%d-%H%M%S")
    run_name = f'{configs.method}_{run_date}'
    if configs.experiment.get('run_name', None) is not None:
        run_name = f"{configs.experiment.run_name}_{run_date}"
    print_helper(f"Starting run at {run_date}")
    summary_path = os.path.join(configs.common.basepath, configs.experiment.summary_path)
    os.makedirs(summary_path, exist_ok=True)
    OmegaConf.save(configs.fullconfig, os.path.join(summary_path, f'{run_name}_config.yaml'))

    # Create the robot
    robot = create_robot(
        robot_config=configs.robot,
        common_config=configs.common
    )
    joint_radius_override = configs.robot.get('joint_radius_override', None)
    print_helper(f"Created robot {robot.name}")

    # Create the planner and setup some viz stuff
    viz = None
    if configs.method =='armtd':
        planner, planner_info = setup_armtd(
            robot,
            device=configs.common.device,
            dtype=configs.common.dtype,
            **configs.planner
        )
        if configs.common.reachset_viz:
            viz = FOViz(planner, plot_full_set=True, t_full=T_FULL)
    elif configs.method == 'sparrows':
        planner, planner_info = setup_sparrows(
            robot,
            device=configs.common.device,
            dtype=configs.common.dtype,
            joint_radius_override=joint_radius_override,
            **configs.planner
        )
        if configs.common.reachset_viz:
            viz = SpherePlannerViz(planner, plot_full_set=True, t_full=T_FULL)
    elif configs.method == 'splanning':
        planner, planner_info = setup_splanner(
            robot,
            device=configs.common.device,
            dtype=configs.common.dtype,
            joint_radius_override=joint_radius_override,
            **configs.planner
        )
        if configs.common.reachset_viz:
            viz = SpherePlannerViz(planner, plot_full_set=True, t_full=T_FULL)
    else:
        raise ValueError(f"Unknown method {configs.method}.")
    planner_settings_export = convert_dict_to_dict_list(planner_info)
    print_helper(f"Created planner {configs.method} with settings {planner_info}")
    # Make sure the right pyopengl platform is set for video rendering
    if configs.common.video:
        import platform
        if platform.system() == "Linux":
            os.environ['PYOPENGL_PLATFORM'] = 'egl'
        video_name = f'{robot.name}_' + ('reachset_' if configs.common.reachset_viz else '')
        video_name += run_date

    # validate data paths
    if not os.path.isdir(configs.experiment.scenes_path):
        raise FileNotFoundError(f"Invalid scenes path: {configs.experiment.scenes_path}. "
                                "Please follow the instructions in the readme to download the data and configure paths.")

    # Get the experiment list
    experiment_list = build_experiment_list(
        basepath=configs.common.basepath,
        scenes_path=configs.experiment.scenes_path,
        results_path=configs.experiment.results_path,
        selected_scenarios=configs.experiment.get('selected_scenarios', None)
    )
    print_helper(f"Found {len(experiment_list)} scenarios to run in {configs.experiment.scenes_path}")
    
    ## Run the experiments
    persistant_env = None
    all_stats = []
    for experiment in tqdm(experiment_list, dynamic_ncols=True, desc='Experiments', position=0):
        print_helper(f"Running experiment {experiment.scenario_name}")
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        if configs.method == "splanning":
            alpha = configs.planner.constraint_alpha
            beta = configs.planner.constraint_beta
            logdir = os.path.join(experiment.result_path, configs.method, f"alpha_{alpha}_beta_{beta}")
        else:
            logdir = os.path.join(experiment.result_path, configs.method)
        os.makedirs(logdir, exist_ok=True)
        # Load / create the environment
        env = setup_environment(
            env_filepath=experiment.scene_file,
            robot=robot.urdf,
            max_steps=configs.experiment.num_steps,
            persistant_env=persistant_env,
            check_self_collision=configs.experiment.env_self_collisions,
            visualize=configs.common.visualize
        )
        persistant_env = env
        if viz is not None and 'reachset' not in env.render_callbacks:
            env.add_render_callback('reachset', viz.render_callback, needs_time=False)

        # Update the splanner planner with the right splats
        if configs.method == 'splanning':
            print_helper(f"Setting splats for {configs.method} and warming up the constraints")
            splat_path, kwargs = get_splat_file(experiment, configs.splats, configs.common.basepath)
            splat_info = splanner_setsplat(planner, splat_path, **kwargs)
            planner_settings_export['splats'] = convert_dict_to_dict_list(splat_info)
        
        # setup the waypoint generator and the goals
        obs = env.get_observations()
        waypoint_generator = GoalWaypointGenerator(obs['qgoal'], planner.osc_rad*3)
        
        # setup the video
        if configs.common.video:
            video_path = os.path.join(logdir, video_name+'.mp4')
            os.makedirs(os.path.dirname(video_path), exist_ok=True)
            video_recorder = FullStepRecorder(env, path=video_path)

        # Planning Loop
        print_helper(f"Starting planning loop for {experiment.scenario_name}")
        stuck = False
        force_fail_safe = False
        log_data = LogData()
        log_data.trajectory_dt = env.t_step / env.timestep_discretization
        for _ in tqdm(range(configs.experiment.num_steps), dynamic_ncols=True, desc='Planning', position=1, leave=False):
            # Observations and waypoint
            qpos, qvel = obs['qpos'], obs['qvel']
            obstacles = (np.asarray(obs['obstacle_pos']), np.asarray(obs['obstacle_size']))
            waypoint = waypoint_generator.get_waypoint(qpos, qvel)

            # Run the planner
            ts = time.time()
            ka, flag, planner_stat = planner.plan(
                qpos,
                qvel,
                waypoint,
                obstacles,
                time_limit=configs.experiment.step_timelimit,
                t_final_thereshold=configs.experiment.t_final_threshold,
            )
            t_elasped = time.time()-ts

            # update ka and the viz, and break if stuck
            if flag != 0:
                if viz is not None:
                    viz.set_ka(None)
                if stuck:
                    log_data.result = "STUCK"
                    break
                ka = (0 - qvel)/(T_FULL - T_PLAN)
                stuck = True
            elif force_fail_safe and configs.experiment.force_failsafe:
                if viz is not None:
                    viz.set_ka(None)
                ka = (0 - qvel)/(T_FULL - T_PLAN)
                force_fail_safe = False
                stuck = False
            else:
                if viz is not None:
                    viz.set_ka(ka)
                force_fail_safe = configs.experiment.force_failsafe and (np.sqrt(planner.final_cost) < env.goal_threshold)
                stuck = False
            
            # Save the planner stats
            log_data.planner_times.append(t_elasped)
            for key in planner_stat:
                if planner_stat[key] is None:
                    continue
                stats = log_data.planner_stats.get(key, [])
                stats += planner_stat[key] if isinstance(planner_stat[key], list) else [planner_stat[key]]
                log_data.planner_stats[key] = stats
            
            # Step the environment and record
            obs, reward, _, info = env.step(ka, save_full_trajectory=configs.experiment.save_trajectories)
            if configs.common.video:
                video_recorder.capture_frame()
            elif configs.common.visualize:
                env.render()
            
            # Save the environment results
            log_data.k.append(ka)
            log_data.q.append(obs['qpos'])
            log_data.qd.append(obs['qvel'])
            log_data.qdd.append(obs['last_action'])
            log_data.flag.append(flag)
            if configs.experiment.save_trajectories:
                log_data.trajectory.append(info['full_trajectory'])
            
            # Final results (NOTE: STUCK is computed inline with ka recomputation)
            if 'collision_info' in info and info['collision_info']['in_collision']:
                log_data.result = "COLLISION"
                log_data.collision_count += 1

            if reward and log_data.result != "COLLISION":
                log_data.result = "SUCCESS"
                break
        if torch.cuda.is_available():
            peak_gpu_mem = torch.cuda.max_memory_allocated()
            log_data.peak_gpu_memory = peak_gpu_mem
            
        # Clean up viz and video
        if configs.common.video:
            video_recorder.close()
        if configs.common.visualize:
            print_helper("Press enter to continue")
            env.spin(wait_for_enter=True)
        
        print_helper(f"Finished planning for {experiment.scenario_name} with result {log_data.result}")
        # Process the log data and save
        if configs.experiment.save_trajectories:
            log_data.trajectory_to_csv(os.path.join(logdir, f"traj_{run_date}.csv"))
        stats_str = log_data.print_planning_stats(prefix=run_date, print_to_stdout=False, save_path=os.path.join(logdir, "stats.txt"))
        if configs.common.verbose:
            tqdm.write(stats_str)

        export_dict = log_data.convert_to_dict_list(ignore_keys=['trajectory'])
        with open(os.path.join(logdir, f"results_{run_date}.json"), 'w') as f:
            json.dump(export_dict, f, indent=2)
        
        with open(os.path.join(logdir, f"planner_settings_{run_date}.json"), 'w') as f:
            json.dump(planner_settings_export, f, indent=2)

            
        # Consolidate stats
        all_stats.append({
            'scenario': experiment.scenario_name,
            'result': log_data.result,
            'collision_count': log_data.collision_count,
            'num_steps': log_data.num_steps,
            'planner_stats': log_data.planner_stats,
            'planner_times': log_data.planner_times,
            'time': sum(log_data.planner_times),
        })
    
    # Create the summary
    # also remove splats from planner settings if present
    planner_settings_export.pop('splats', None)
    planner_stats_summary = {}
    for scene in all_stats:
        for k, v in scene['planner_stats'].items():
            total_list = planner_stats_summary.get(k, [])
            total_list += v
            planner_stats_summary[k] = total_list
    for key, comb_list in planner_stats_summary.items():
        planner_stats_summary[key] = {
            'mean': np.mean(comb_list),
            'std': np.std(comb_list),
            'max': float(np.max(comb_list)),
            'min': float(np.min(comb_list)),
        }
    planning_times_all = [val for scene in all_stats for val in scene['planner_times']]
    planning_times_scenes_all = [s['time'] for s in all_stats]
    success_planning_times = [val for scene in all_stats if scene['result'] == 'SUCCESS' for val in scene['planner_times']]
    trial_data = [{k: v for k, v in s.items() if k not in ['planner_stats', 'planner_times']} for s in all_stats]
    summary_out = {
        'method': configs.method,
        'num_scenarios': len(all_stats),
        'num_success_scene': sum(1 for s in all_stats if s['result'] == 'SUCCESS'),
        'num_collision_scene': sum(1 for s in all_stats if s['result'] == 'COLLISION'),
        'num_stuck_scene': sum(1 for s in all_stats if s['result'] == 'STUCK'),
        'mean_scene_times': np.mean(planning_times_scenes_all),
        'std_scene_times': np.std(planning_times_scenes_all),
        'mean_scene_times_success': np.mean(success_planning_times),
        'std_scene_times_success': np.std(success_planning_times),
        'mean_planning_times': np.mean(planning_times_all),
        'std_planning_times': np.std(planning_times_all),
        'mean_planning_times_success': np.mean(success_planning_times),
        'std_planning_times_success': np.std(success_planning_times),
        'total_planning_times': sum(planning_times_all),
        'total_planning_times_success': sum(success_planning_times),
        'num_incomplete': sum(1 for s in all_stats if s['result'] == 'INCOMPLETE'),
        'num_steps': sum(s['num_steps'] for s in all_stats),
        'num_steps_success': sum(s['num_steps'] for s in all_stats if s['result'] == 'SUCCESS'),
        'planner_stats_summary': planner_stats_summary,
        'trial_data': trial_data,
        'planner_settings': planner_settings_export,
    }

    # Save the summary and configs to rerun
    with open(os.path.join(summary_path, f'{run_name}.json'), 'w') as f:
        json.dump(summary_out, f, indent=2)


if __name__ == '__main__':
    main()
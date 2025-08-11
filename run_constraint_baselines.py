from collision_comparisons.config import process_collision_config
from tqdm import tqdm
import time
import os
from omegaconf import OmegaConf

from util import (
    create_robot,
    build_experiment_list,
    sampling_config_generator,
    get_splat_file,
    get_nerfstudio_config
)
from collision_comparisons.sphere_fk import SphereFK
from collision_comparisons.configuration_sampling import GroundtruthConfigurationSampler
from collision_comparisons.splat_constraints import SplatConstraints
import torch
import numpy as np
import random

torch.set_float32_matmul_precision('high')

def main():
    # Process file configs and command line arguments
    configs = process_collision_config()
    torch.manual_seed(configs.common.seed)
    np.random.seed(configs.common.seed)
    random.seed(configs.common.seed)

    # helper function for printing
    def print_helper(*args, **kwargs):
        if configs.common.verbose:
            tqdm.write(*args, **kwargs)

    # Get the experiment list
    experiment_list = build_experiment_list(
        basepath=configs.common.basepath,
        scenes_path=configs.common.scenes_path,
        results_path=configs.common.datapath,
        selected_scenarios=configs.common.selected_scenarios
    )
    print_helper(f"Found {len(experiment_list)} scenarios to run in {configs.common.scenes_path}")    
    
    # Date and summary storage
    run_date = time.strftime("%Y%m%d-%H%M%S")
    run_name = f'{configs.common.prefix}_{run_date}'
    if configs.common.prefix is not None:
        run_name = f"{configs.common.prefix}_{run_date}"
    print_helper(f"Starting run at {run_date}")
    summary_path = os.path.join(configs.common.basepath, configs.common.datapath)
    os.makedirs(summary_path, exist_ok=True)
    configs.fullconfig.run_date = run_date # add the run date to the config
    OmegaConf.save(configs.fullconfig, os.path.join(summary_path, f'{run_name}_config.yaml'))

    # Create the robot and sphere FK
    robot = create_robot(
        robot_config=configs.robot,
        common_config=configs.common
    )
    sphere_fk = SphereFK(
        robot=robot,
        **configs.robot.get('sphere_fk_params', dict())
    )
    print_helper(f"Created robot {robot.name} and setup sphere FK as {sphere_fk}.")

    # Sample and generate the waypoints and spheres for the selected scenes
    sphere_sampler_loader = GroundtruthConfigurationSampler(
        sphere_fk=sphere_fk,
        **configs.sample_generation
    )
    for experiment in tqdm(experiment_list, dynamic_ncols=True, desc='Samples & GT Precomp', position=0):
        os.makedirs(experiment.result_path, exist_ok=True)
        # Load or sample and generate the waypoints and spheres for the selected scene
        sphere_sampler_loader.sample_load_environment(
            experiment.scene_file,
            exp_datapath=experiment.result_path,
            prefix=configs.common.prefix,
            run_date=run_date,
        )
    print_helper("Finished all samples and groundtruth precomputation.")

    # start constraints
    splat_constraints = None
    if configs.splanning.skip:
        print_helper("Skipping SPLANNING constraints")
    else:
        if splat_constraints is None:
            splat_constraints = SplatConstraints(
                device=configs.common.device,
                dtype=configs.common.dtype
            )
        alphas = sampling_config_generator(
            configs.splanning.alpha_sampling
        )
        alphas = torch.tensor(alphas, dtype=configs.common.dtype, device=configs.common.device)
        for experiment in tqdm(experiment_list, dynamic_ncols=True, desc='SPLANNING Constraints', position=0):
            os.makedirs(experiment.result_path, exist_ok=True)
            # Load or sample and generate the waypoints and spheres for the selected scene
            sphere_sampler_loader.sample_load_environment(
                experiment.scene_file,
                exp_datapath=experiment.result_path,
                prefix=configs.common.prefix,
                run_date=run_date,
            )
            test_spheres, gt_sphere_coll, gt_coll = sphere_sampler_loader.get_gt_spheres(configs.splanning.gt_set)

            # Prepare the constraint
            print_helper(f"Setting splats for {experiment.scenario_name}, SPLANNING")
            splat_path, kwargs = get_splat_file(experiment, configs.splanning.splats, configs.common.basepath)
            splat_constraints.setup_splat(splat_path, **kwargs)
            print_helper(f"Generating constraints values...")
            sphere_collision_probabilities = splat_constraints.splanning_gsplats_constraint(
                centers=test_spheres[0],
                radii=test_spheres[1],
                alphas=alphas
            )

            # Export values for separate analysis
            alphas_out = alphas.cpu().numpy()
            sc_lhs = sphere_collision_probabilities.cpu().numpy()
            cc_lhs = sphere_collision_probabilities.sum(-1).cpu().numpy()
            export = {
                'alphas': alphas_out,
                'sphere_collisions_lhs': sc_lhs,
                'config_collisions_lhs': cc_lhs,
                'sphere_collisions': sc_lhs > alphas_out[:,None,None],
                'config_collisions': cc_lhs > alphas_out[:,None],
                'gt_sphere_collisions': gt_sphere_coll,
                'gt_config_collisions': gt_coll,
            }

            # Save the results
            save_path = os.path.join(experiment.result_path, f'{configs.common.prefix}_splanning_{run_date}.npz')
            np.savez(save_path, **export)
            print_helper(f"Saved splanning_constraint results to {save_path}")

    if configs.splat_nav.skip:
        print_helper("Skipping Splat-Nav constraints")
    else:
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        if splat_constraints is None:
            splat_constraints = SplatConstraints(
                device=configs.common.device,
                dtype=configs.common.dtype
            )
        sigma_levels = sampling_config_generator(
            configs.splat_nav.sigma_level_set_sampling
        )
        sigma_levels = torch.tensor(sigma_levels, dtype=configs.common.dtype, device=configs.common.device)
        for experiment in tqdm(experiment_list, dynamic_ncols=True, desc='Splat-Nav Constraints', position=0):
            os.makedirs(experiment.result_path, exist_ok=True)
            # Load or sample and generate the waypoints and spheres for the selected scene
            sphere_sampler_loader.sample_load_environment(
                experiment.scene_file,
                exp_datapath=experiment.result_path,
                prefix=configs.common.prefix,
                run_date=run_date,
            )
            test_spheres, gt_sphere_coll, gt_coll = sphere_sampler_loader.get_gt_spheres(configs.splat_nav.gt_set)

            # Prepare the constraint
            print_helper(f"Setting splats for {experiment.scenario_name}, Splat-Nav")
            splat_path, kwargs = get_splat_file(experiment, configs.splat_nav.splats, configs.common.basepath)
            splat_constraints.setup_splat(splat_path, **kwargs)
            print_helper(f"Generating constraints values...")
            collisions = splat_constraints.splatnav_gsplats_constraint(
                centers=test_spheres[0],
                radii=test_spheres[1],
                sigma_levels=sigma_levels
            )

            # Export values for separate analysis
            export = {
                'sigma_levels': sigma_levels.cpu().numpy(),
                'sphere_collisions': collisions.cpu().numpy(),
                'config_collisions': collisions.sum(-1).cpu().numpy(),
                'gt_sphere_collisions': gt_sphere_coll,
                'gt_config_collisions': gt_coll,
            }

            # Save the results
            save_path = os.path.join(experiment.result_path, f'{configs.common.prefix}_splatnav_{run_date}.npz')
            np.savez(save_path, **export)
            print_helper(f"Saved splatnav_constraint results to {save_path}")

    if configs.catnips.skip:
        print_helper("Skipping CATNIPS constraints")
    else:
        from concurrent.futures import ProcessPoolExecutor
        from functools import partial
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        # Setup sigma param sweep
        sigma_levels = sampling_config_generator(
            configs.catnips.sigmas_sampling
        )
        if configs.catnips.get('gpu_accel', None) is not None:
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                available_gpus = [int(g) for g in os.environ["CUDA_VISIBLE_DEVICES"].split(',')]
            else:
                available_gpus = list(range(torch.cuda.device_count()))
            visible_gpus = configs.catnips.gpu_accel.get('visible_gpus', None)
            if visible_gpus is None:
                visible_gpus = available_gpus
            else:
                visible_gpus = [available_gpus[g] for g in visible_gpus]
            visible_gpus = [str(g) for g in visible_gpus]
            nprocs_per_gpu = configs.catnips.gpu_accel.get('nprocs_per_gpu', 1)
            nprocs = len(visible_gpus) * nprocs_per_gpu
            print_helper(f"Using {nprocs} processes for CATNIPS constraints")
            executor = ProcessPoolExecutor(max_workers=nprocs, mp_context=torch.multiprocessing.get_context('spawn'))
            gpu_list = visible_gpus * nprocs_per_gpu
        else:
            executor = ProcessPoolExecutor(max_workers=1, mp_context=torch.multiprocessing.get_context('spawn'))
            visible_gpus = None
            gpu_list = [None]
        executor.map(configure_gpu, gpu_list)
        verbose_subprocess = configs.catnips.verbose_subprocess
        # Setup common CATNIPS config
        catnips_configs = dict(configs.catnips.catnips_base)
        Vaux = None
        if 'Vaux' in configs.catnips and configs.catnips.Vaux is not None:
            Vaux = configs.catnips.Vaux
            catnips_configs['Aaux'] = Vaux / catnips_configs['dt']
        
        for experiment in tqdm(experiment_list, dynamic_ncols=True, desc='CATNIPS Constraints', position=0):
            # gc.collect()
            # torch.cuda.empty_cache()
            os.makedirs(experiment.result_path, exist_ok=True)
            sphere_sampler_loader.sample_load_environment(
                experiment.scene_file,
                exp_datapath=experiment.result_path,
                prefix=configs.common.prefix,
                run_date=run_date,
            )
            test_spheres, gt_sphere_coll, gt_coll = \
                sphere_sampler_loader.get_gt_spheres(configs.catnips.gt_set)

            # get the relevant config for each nerf
            nf_cfg = get_nerfstudio_config(experiment, configs.catnips.nerfs, configs.common.basepath)

            # if we're using the provided settings, Vaux is null and defined indirectly through catnips settings
            if Vaux is None:
                print_helper(f"Testing CATNIPS with provided settings")
                sigma_loop_func = partial(catnips_helper, extras=(catnips_configs, nf_cfg, test_spheres, None, Vaux, verbose_subprocess))
                collisions_out = list(
                    tqdm(executor.map(sigma_loop_func, sigma_levels),
                        dynamic_ncols=True,
                        desc='Sigmas Progress (est.)',
                        position=1,
                        total=len(sigma_levels)
                    ))

                collisions_out = np.stack(collisions_out, axis=0)
                # Export values for separate analysis
                export = {
                    'sigma_levels': sigma_levels,
                    'sphere_collisions': collisions_out,
                    'config_collisions': np.sum(collisions_out, axis=-1),
                    'gt_sphere_collisions': gt_sphere_coll,
                    'gt_config_collisions': gt_coll,
                }
                # import open3d as o3d
                # collision_spheres = test_spheres[0][collisions_out[0]]
                # pcd = o3d.geometry.PointCloud()
                # pcd.points = o3d.utility.Vector3dVector(collision_spheres)
                # o3d.io.write_point_cloud(f"collision.pcd", pcd)
                # safe_spheres = test_spheres[0][~collisions_out[0]]
                # pcd = o3d.geometry.PointCloud()
                # pcd.points = o3d.utility.Vector3dVector(safe_spheres)
                # o3d.io.write_point_cloud(f"safe.pcd", pcd)

                # Save the results
                save_path = os.path.join(experiment.result_path, f'{configs.common.prefix}_catnips_{run_date}.npz')
                np.savez(save_path, **export)
                print_helper(f"Saved catnips_constraint results for the provided settings to {save_path}")
                continue

            # otherwise iterate over the Nmax_aux_list
            for Nmax_aux in tqdm(configs.catnips.Nmax_aux_list, dynamic_ncols=True, desc='Nmax_aux', position=1):
                print_helper(f"Testing CATNIPS for Nmax_aux={Nmax_aux}")
                sigma_loop_func = partial(catnips_helper, extras=(catnips_configs, nf_cfg, test_spheres, Nmax_aux, Vaux, verbose_subprocess))
                collisions_out = list(
                    tqdm(executor.map(sigma_loop_func, sigma_levels),
                        dynamic_ncols=True,
                        desc='Sigmas Progress (est.)',
                        position=2,
                        total=len(sigma_levels)
                    ))

                collisions_out = np.stack(collisions_out, axis=0)
                # Export values for separate analysis
                export = {
                    'sigma_levels': sigma_levels,
                    'sphere_collisions': collisions_out,
                    'config_collisions': np.sum(collisions_out, axis=-1),
                    'gt_sphere_collisions': gt_sphere_coll,
                    'gt_config_collisions': gt_coll,
                }

                # Save the results
                save_path = os.path.join(experiment.result_path, f'{configs.common.prefix}_catnips_{Nmax_aux}_{run_date}.npz')
                np.savez(save_path, **export)
                print_helper(f"Saved catnips_constraint results for Nmax_aux={Nmax_aux} to {save_path}")
        executor.shutdown()

    print_helper("Finished all constraints.")


def catnips_helper(sigma, extras):
    import os, multiprocessing, sys
    catnips_configs, nf_cfg, test_spheres, Nmax_aux, Vaux, verbose_subprocess = extras
    if multiprocessing.parent_process() is not None and not verbose_subprocess:
        # TODO redirect properly
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
    from collision_comparisons.catnips.catnips_constraint import CatnipsConstraint
    from collision_comparisons.catnips.nerf import NeRFWrapper
    import gc
    nerf_wrapper = NeRFWrapper(nf_cfg)
    catnips_configs['sigma'] = sigma
    if Vaux is not None:
        catnips_configs['Vmax'] = Nmax_aux * Vaux
    catnips_constraint = CatnipsConstraint(nerf_wrapper, catnips_configs, test_spheres[1])
    collisions = catnips_constraint.constraint(
        centers=test_spheres[0],
    )
    # catnips_constraint._catnips.save_purr(
    #     f"test_{sigma}.ply", catnips_constraint.nerf_wrapper.transform.detach().cpu().numpy(), catnips_constraint.nerf_wrapper.scale)
    del nerf_wrapper, catnips_constraint
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return collisions


def configure_gpu(gpu_choice):
    if gpu_choice is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_choice
        import time
        # Hacky way to force the assignment of the right cuda device
        time.sleep(5)
    

if __name__ == '__main__':
    main()

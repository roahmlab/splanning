from collision_comparisons.config import process_collision_config
from tqdm import tqdm
import time
import os
from omegaconf import OmegaConf, ListConfig

from util import (
    create_robot,
    build_experiment_list,
    sampling_config_generator,
    get_splat_file,
    convert_dict_to_dict_list,
    get_default_with_warning,
    resolve_incomplete_filename,
)
from collision_comparisons.sphere_fk import SphereFK
from collision_comparisons.configuration_sampling import GroundtruthConfigurationSampler
from collision_comparisons.splat_constraints import SplatConstraints
from collision_comparisons.analysis import do_confusion, calculate_precision_recall
import torch
import numpy as np
import random
import json
torch.set_float32_matmul_precision('high')


DEFAULT_BASE_CONFIG = 'configs/collisions_baselines/analysis.yml'
def main():
    cli_conf = OmegaConf.from_cli()
    # Base is loaded, then overlay, then cli is added on top
    # If rerun is specified, then rerun is treated as base, overlay is errored, and cli is added on top
    base_conf_path = cli_conf.pop('config', DEFAULT_BASE_CONFIG)

    # Merge the configs
    config = OmegaConf.load(base_conf_path)
    config = OmegaConf.merge(config, cli_conf)

    basepath = get_default_with_warning(config, 'basepath', None)
    if basepath is None or basepath == '':
        basepath = os.getcwd()
    
    # Make a functional so we can load this from other configs used to construct
    experiment_list_functional = lambda datapath: build_experiment_list(
        basepath=basepath,
        scenes_path=config.scenes_path,
        results_path=datapath,
        selected_scenarios=config.selected_scenarios
    )

    default_run_config = config.run_config
    
    collected_data = {}

    for dataset_name, dataset_cfg in config.trial_data.items():
        if dataset_cfg.include == False:
            continue
        dataset_run_config = get_default_with_warning(dataset_cfg, 'run_config', None)
        if dataset_run_config is None:
            dataset_run_config = default_run_config
        if isinstance(dataset_run_config, (list, ListConfig)):
            dataset_run_config = [resolve_incomplete_filename(os.path.join(basepath,rc)) for rc in dataset_run_config]
            dataset_run_config = [OmegaConf.load(rc) for rc in dataset_run_config]
            dataset_exp_list = [
                (exp, rc.prefix, rc.run_date)
                for rc in dataset_run_config
                for exp in experiment_list_functional(rc.datapath)
            ]
        else:
            dataset_run_config = resolve_incomplete_filename(os.path.join(dataset_run_config))
            dataset_run_config = OmegaConf.load(dataset_run_config)
            dataset_exp_list = [(exp, dataset_run_config.prefix, dataset_run_config.run_date) for exp in experiment_list_functional(dataset_run_config.datapath)]
        
        tp, fp, tn, fn = 0, 0, 0, 0
        parameter = None
        for experiment, prefix, run_date in dataset_exp_list:
            loadfile_path = os.path.join(experiment.result_path, f'{prefix}_{dataset_cfg.method}_{run_date}.npz')
            if not os.path.exists(loadfile_path):
                import warnings
                warnings.warn(f"Skipping {experiment.scenario_name} as file {loadfile_path} does not exist.")
                continue
            data = np.load(loadfile_path)
            if parameter is None:
                parameter = data[dataset_cfg.parameter_key]
            test_val = data[dataset_cfg.comparison_key]
            gt_val = data[dataset_cfg.gt_key]
            confusion = do_confusion(gt_val, test_val)
            tp = tp + confusion.true_positives
            fp = fp + confusion.false_positives
            fn = fn + confusion.false_negatives
            tn = tn + confusion.true_negatives
            del data
        precision, recall = calculate_precision_recall(tp, fp, fn)

        collected_data[dataset_name] = {
            'param_name': dataset_cfg.parameter_key,
            'param_values': parameter,
            'precision': precision,
            'recall': recall,
            'confusion': {
                'true_positives': tp,
                'false_positives': fp,
                'false_negatives': fn,
                'true_negatives': tn
            }
        }

    # Save the results to json
    if config.export.data_filename is not None:
        out_dict = convert_dict_to_dict_list(collected_data)
        out_path = os.path.join(basepath, config.export.data_filename)
        with open(out_path, 'w') as f:
            json.dump(out_dict, f)
        print(f"Saved results to {out_path}")

    # Also plot if plot key exists
    if 'plot' in config.export and config.export.plot.enabled:
        import matplotlib.pyplot as plt
        import copy
        all_lines = []
        for dataset_name, data in collected_data.items():
            styles = config.export.plot.styles.get(dataset_name, {})
            if OmegaConf.is_config(styles):
                styles = OmegaConf.to_object(styles)
            highlight = styles.pop('highlight', None)
            plt.plot(
                data['recall'],
                data['precision'],
                **styles)
            all_lines.append(((data['recall'],data['precision']), copy.deepcopy(styles)))
            if highlight is not None:
                idx = highlight.pop('idx')
                styles.update(highlight)
                styles['label'] = '_' + styles['label']
                plt.plot(
                    data['recall'][idx],
                    data['precision'][idx],
                    **styles
                )
                all_lines.append(((data['recall'][idx],data['precision'][idx]), copy.deepcopy(styles)))
        plt.legend()
        plt.legend(loc='best', fontsize=10)
        ax_main = plt.gca()
        from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
        # inset_ax = zoomed_inset_axes(ax_main, zoom=3, log='lower left', borderpad=1.0)
        # for line in ax_main.get_lines():
        #     inset_ax.plot(line.get_xdata(), line.get_ydata(), linestyle=line.get_linestyle(), marker=line.get_marker(), color=line.get_color())

        
        inset = config.export.plot.get('inset', None)
        if inset is not None:
            inset_ax = plt.axes(inset.axes)
            inset_ax.set_xlim(*inset.xlim)
            inset_ax.set_ylim(*inset.ylim)
            inset_ax.grid()
            for line in all_lines:
                inset_ax.plot(*line[0], **line[1])
            mark_inset(ax_main, inset_ax, loc1=2, loc2=4, ec="0.5", fc="None")

        ax = plt.subplot(111)
        # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 0.05), fontsize=4, ncols=1)
        plt.grid()
        plt.xlabel('Recall', fontsize=20)
        plt.ylabel('Precision', fontsize=20)
        plt.xlim([0,1.1])
        plt.ylim([0,1.1])
        # plt.axis('equal')
        # plt.title('Collision Precision-Recall', fontsize=20)
        plt.subplots_adjust(bottom=0.25)

        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                        box.width, box.height * 0.9])

        # Put a legend below current axis
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
                fancybox=True, shadow=False, ncol=3)
        
        if config.export.plot.pdf is not None:
            plt.savefig(config.export.plot.pdf, format='pdf')
        if config.export.plot.eps is not None:
            plt.savefig(config.export.plot.eps, format='eps')
        if config.export.plot.png is not None:
            plt.savefig(config.export.plot.png, format='png')
        

if __name__ == '__main__':
    main()

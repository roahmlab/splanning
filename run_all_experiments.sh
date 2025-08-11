#!/bin/bash

n_obs_list=(10 20 40)
methods=("armtd" "sparrows" "splanning")
risk_thresholds=(0.01 0.025 0.0375 0.05 0.1) 

for n_obs in "${n_obs_list[@]}"; do
    for method in "${methods[@]}"; do
        if [ "$method" == "splanning" ]; then
            for risk in "${risk_thresholds[@]}"; do
                echo "Running experiment with method=$method, n_obs=$n_obs, constraint_alpha=$risk, constraint_beta=$risk"
                python3 run_experiments.py experiment.method="$method" \
                                           base_config=./configs/experiments.yml \
                                           experiment.selected_scenarios="${n_obs}obs" \
                                           splanning.constraint_alpha="$risk" \
                                           splanning.constraint_beta="$risk" \
                                           experiment.run_name="splanning_${n_obs}obs_risk${risk}"
            done
        else
            echo "Running experiment with method=$method and n_obs=${n_obs}"
            python3 run_experiments.py experiment.method="$method" \
                                       base_config=./configs/experiments.yml \
                                       experiment.selected_scenarios="${n_obs}obs" \
                                       experiment.run_name="${method}_${n_obs}obs"
        fi
    done
done
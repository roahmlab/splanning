# SPLANNING
## Let's Make a Splan: Risk-Aware Trajectory Optimization in a Normalized Gaussian Splat

[//]: <[Project Page](https://roahmlab.github.io/RDF/) | [Paper](https://arxiv.org/abs/2302.07352) | [Dataset](https://drive.google.com/drive/folders/1sxRCtuAwi2Ua5BIVX0fLqOBlb95PcFN0?usp=share_link)>

## Introduction

This repository contains an implementation of the _SPLANNING_ algorithm from the paper _Let's Make a Splan: Risk-Aware Trajectory Optimization in a Normalized Gaussian Splat_. 

<details>
  <summary>Click to Expand Abstract</summary>
Neural Radiance Fields and Gaussian Splatting have recently transformed computer vision by enabling photo-realistic representations of complex scenes. However, they have seen limited application in real-world robotics tasks such as trajectory optimization. This is due to the difficulty in reasoning about collisions in radiance models and the computational complexity associated with operating in dense models. This paper addresses these challenges by proposing SPLANNING, a risk-aware trajectory optimizer operating in a Gaussian Splatting model. This paper first derives a method to rigorously upper-bound the probability of collision between a robot and a radiance field. Then, this paper introduces a normalized reformulation of Gaussian Splatting that enables efficient computation of this collision bound. Finally, this paper presents a method to optimize trajectories that avoid collisions in a Gaussian Splat. Experiments show that SPLANNING outperforms state-of-the-art methods in generating collision-free trajectories in cluttered environments. The proposed system is also tested on a real-world robot manipulator. A project page is available at [roahmlab.github.io/splanning](https://roahmlab.github.io/splanning).
</details>

<br>

This repository consists of code needed to run the SPLANNING planner and reproduce the experiments in the paper.
A separate repository at [roahmlab/normalized_splatting](https://github.com/roahmlab/normalized_splatting) contains the Normalized 3DGS code.

## Installation & Setup

### Conda Environment

tl;dr: To set up the conda environment, run this (ideally with a version of conda using the [libmamba solver](https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community) for speed).

```bash
cd /path/to/repo
conda create -n splanning
conda activate splanning
conda config --env --set channel_priority flexible
conda env update -n splanning -f environment.yaml
```

<details>
<summary> Click to expand details on Conda Installation</summary>
Python dependencies can be installed directly from `environment.yaml` using [conda](https://www.anaconda.com) environments.
We also provide instructions for other systems below.
We recommend using a version of conda with the [libmamba solver](https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community).

Newer versions of conda also use a strict channel priority, which will need to be set flexible.
```bash
conda create -n splanning
conda activate splanning
conda config --env --set channel_priority flexible
conda env update -n splanning -f environment.yaml
```

If your system is already setup with flexible channel priority, you can configure as follows:
```bash
conda env create --file environment.yaml
conda activate splanning
```


Note that the environment includes:
- [zonopy](https://github.com/roahmlab/zonopy) which provides functionalities used for reachability analysis
- [zonopy-robots](https://github.com/roahmlab/zonopy-robots) which provides functionalities for specifying and loading robots
</details>

### IPOPT and HSL Setup
To run our algorithms you need a Coin HSL license.

Navigate to [https://licences.stfc.ac.uk/product/coin-hsl](https://licences.stfc.ac.uk/product/coin-hsl), make an account, and order a free Academic License. This is generally approved for academic users within a couple days.

Once your license is approved, sign into your account at [https://licences.stfc.ac.uk/product/coin-hsl](https://licences.stfc.ac.uk/product/coin-hsl), select "My Account" in the top-right, and click on "My Downloads". Alternatively, follow [this link](https://licences.stfc.ac.uk/account/downloads) once you have signed in. 

Download the `CoinHSL 2023.11.17 (tarball)` file to some location `/path/to/coinhsl-2023.11.17.tar.gz`.

From a terminal, run the following:

```bash
conda activate splanning #if not already activated
export COINHSL_PATH=/path/to/coinhsl-2023.11.17.tar.gz # wherever you downloaded this to
bash setup_ipopt.sh
```

This script will walk you through the installation.

### Optional Extra Dependencies

For advanced users who wish to re-compute reachable sets or a gaussian error table (used for culling far-away Gaussians), you will need MATLAB, CORA, and WolframScript. Most users can neglect this and use our provided values.

<details>
<summary>Click to expand details on Optional Extra Dependencies</summary>

[MATLAB](https://matlab.mathworks.com) and [CORA 2021](https://tumcps.github.io/CORA/) are used to compute Joint Reachable Set in `forward_occupancy/jrs_trig/gen_jrs_trig` with the provided MATLAB scripts.

[WolframScript](https://www.wolfram.com/wolframscript/index.php.en) is used to generate `gaussian_table.csv` in `planning/splanning/` with the provided `compute_gaussian_error.wls` script.

</details>


## Reproducing Results

### Downloading Dataset

Download the dataset from [DeepBlue](https://deepblue.lib.umich.edu/data/concern/data_sets/c534fp99m?locale=en). You may need to use Globus following the instructions on the page.

If you prefer not to download the full dataset (which includes additional data for running baselines), you may instead download only the following subdirectories:

```
scenes/
models/normalized_3dgs/
```

After downloading, ensure the directory structure at least contains the following:
```
data_root/
├── scenes/
└── models/
    └── normalized_3dgs/
```


You then need to point SPLANNING to the location you have downloaded the file to. To do this, run:

```bash
python3 configs/register_data_location.py /path/to/data_root 
```

If `/path/to/data_root` does not contain subdirectories `models/normalized_3dgs` and `scenes`, this script will throw an error!

### Running Planning Experiments

There are 300 total scenes organized as follows:
```bash
10obs/
├── scene_0/
├── ...
└── scene_99/
20obs/
├── scene_0/
├── ...
└── scene_99/
40obs/
├── scene_0/
├── ...
└── scene_99/
```

You can run all the scenes or just a subset. For example:

```bash
# Run just one scenario: 10obs/scene_0, and save a video with the reachsets visualized
python3 run_experiments.py experiment.method=splanning experiment.selected_scenarios=10obs/scene_0 video=true reachset_viz=true

# run all the 10-obstacles scenes:
python3 run_experiments.py experiment.method=splanning experiment.selected_scenarios=10obs/ 

# Reproduce the demo vidoes from the website:
python3 run_experiments.py experiment.method=splanning experiment.selected_scenarios=[10obs/scene_72,20obs/scene_16,40obs/scene_71] video=true reachset_viz=true

# Run all 300 scenes without saving any videos
python3 run_experiments.py experiment.method=splanning
```

There are many options available which you can browse in `configs/experiments.yml` and set either by modifying that file or via the command line (using [OmegaConf](https://omegaconf.readthedocs.io/en/2.3_branch/) syntax).

<details>
<summary>Click to expand a summary of the most useful options.</summary>
- `video=true` Will save a video. This makes the script run slower as frames are saved. 
- `visualize=true` will open a visualization window as it runs. This makes the script run slower as the visualization is run.
- `reachset_viz=true` will add visualization of the robot's Spherical Forward Occuapncy to the visualization window and/or logged video.
- `experiment.step_timelimit=VAL` can be used to extend the amount of time the optimizer has to find a solution. We use 0.5s to ensure online operation, but on less powerful hardware you may need to increase this.
- `experiment.method=<splanning|armtd|sparrows>` selects the planner to run.
- `experiment.results_path` sets the location to save the individual planning results for each trial.
- `experiment.summary_path` configures where to write the summary of all trials that are run.
</details>



### Comparing to Baselines

`./run_all_experiments.sh` will run all planning experiments for SPLANNING, SPARROWS, and ARMTD, along with the varying risk thresholds for SPLANNING.

### Generating Joint Reachable Sets (JRS)
You may optionally choose to re-generate the robot's Joint Reachable Sets. This should generally not be neccesary, but can be used to re-create our experiments from scratch.

To re-compute the JRS, open this repository in MATLAB and make sure [CORA 2021](https://tumcps.github.io/CORA/) is on your MATLAB path.

From `forward_occupancy/jrs_trig/gen_jrs_trig`, run `create_jrs_trig_tensor.m` so that the JRS files will be saved in `jrs_trig_tensor_saved/`. The variable `d_kai` in `create_jrs_trig_tensor.m` corresponds to the acceleration range in the paper.

**Pre-generated versions** for pi/6, pi/12, and pi/24 can be found [in this drive link here](https://drive.google.com/file/d/1au1xB_Jsue7LZqvOILgt5NiynhmxsgBu/view?usp=drive_link), but the corresponding folders (e.g. `jrs_trig_tensor_saved_pi_24`) will need to be renamed to `jrs_trig_tensor_saved` and moved to the correct location to match the generated output.

### Recreating Constraint Comparisons


#### Setup

CUDA is required (including both runtime and development libraries) to reproduce baseline results. Verify that CUDA is properly installed with nvcc --version. This ensures the full toolkit is available, not just the runtime.
A separete conda environment is recommended for this as `nerfstudio` is required along with `tiny-cuda-nn` for CATNIPS.
We provide a working environment file at `collision_comparisons/environment-baselines.yaml`, which can be setup following the same steps as before:

```bash
conda create -n splanning_baselines
conda activate splanning_baselines
conda config --env --set channel_priority flexible
conda env update -n splanning_baselines -f environment-baselines.yaml
```

__Note: After building the environment you should manually `pip3 install numpy==1.26.2` to avoid issues with Numpy 2__


Due to licensing restrictions, we are unable to distribute the source code required for comparisons with Splat-Nav and CATNIPS. Instead, please navigate to the following three files in this repository and manually populate them with the code from the linked sources provided in each file:
```
splanning/
├── collision_comparisons/
    ├── splat_nav/
        └── intersection_utils.py
    ├── catnips/
        ├── purr.py        
        └── purr_utils.py 
```

In each of those files, instructions are provided to populate them with the correct content.

Finally, to run the collision comparisons, you must download the entire dataset (including all NeRFStudio Models, Normalized 3DGS, and Un-Normalized 3DGS) following the instructions above. Once that is done, be sure to run the `register_data_location.py` script as described above.

#### Running and Visualizing Results

`python3 run_constraint_baselines.py` will run the baselines.
Relevant configuration settings can be found in `configs/collisions_baselines/base.yml`.

`python3 analyze_constraint_baselines.py` will generate the resulting precision-recall plots, with `configs/collisions_baselines/analysis.yml` describing the relevant configuration settings.

## Troubleshooting

#### Any Numpy-Related Error
If you see any error related to numpy, please first try `pip3 install numpy==1.26.4.

#### Import Error: version GLIBCXX_3.4.29' not found
Your gcc is out of date. Install gcc>10. We use gcc11.

## Citation
```bibtex
@article{michauxisaacson2024splanning,
  author={Michaux, Jonathan and Isaacson, Seth and Adu, Challen Enninful and Li, Adam and Swayampakula, Rahul Kashyap and Ewen, Parker and Rice, Sean and Skinner, Katherine A. and Vasudevan, Ram},
  journal={IEEE Transactions on Robotics}, 
  title={Let's Make a Splan: Risk-Aware Trajectory Optimization in a Normalized Gaussian Splat}, 
  year={2025},
  volume={},
  number={},
  pages={1-19},
  keywords={Robots;Collision avoidance;Planning;Neural radiance field;Trajectory optimization;Computational modeling;Geometry;Real-time systems;3D gaussian splatting;collision avoidance;motion and path planning},
  doi={10.1109/TRO.2025.3584559}
}
```





import argparse
import os
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register location of dowloaded SPLANNING data")
    parser.add_argument("data_dir", help="Path to the downloaded SPLANNING data")
    args = parser.parse_args()
    
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.representer.add_representer(
        type(None),
        lambda self, data: self.represent_scalar('tag:yaml.org,2002:null', 'null')
    )

    data_dir = args.data_dir
    if not os.path.isdir(data_dir):
        raise ValueError(f"The directory {data_dir} does not exist, or is not a directory. Please follow the instructions in the README for downloading the data.")
    
    scenes_dir = os.path.join(data_dir, "scenes")
    if not os.path.isdir(scenes_dir):
        raise ValueError(f"The directory {scenes_dir} does not exist, or is not a directory. "
                          "Please follow the instructions in the README for downloading the data. "
                          "Make sure that the path you provide contains subdirectory `scenes`.")
    
    model_dir = os.path.join(data_dir, "models", "normalized_3dgs")

    if not os.path.isdir(model_dir):
        raise ValueError(f"The directory {model_dir} does not exist, or is not a directory. "
                         "Please follow the instructions in the README for downloading the data. "
                         "Make sure that the path you provide contains subdicrectory `models/normalized_3dgs/`.")
    
    
    code_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(code_dir, 'experiments.yml')) as f:
        data = yaml.load(f)
    
    data['splats']['models_path'] = model_dir
    data["experiment"]["scenes_path"] = scenes_dir
    
    with open(os.path.join(code_dir, 'experiments.yml'), 'w') as f:
        yaml.dump(data, f)
        
    
    
    baselines_config = os.path.join(code_dir, 'collisions_baselines', 'base.yml')
    with open(baselines_config) as f:
        baselines_data = yaml.load(f)
        
    baselines_data['scenes_path'] = scenes_dir
    baselines_data['splanning']['splats']['models_path'] = model_dir
    
    nerfstudio_dir = os.path.join(data_dir, "models", "nerfstudio")
    if os.path.isdir(nerfstudio_dir):
        baselines_data['catnips']['nerfs']['models_path'] = nerfstudio_dir
    else:
        print(f"The Nerfstudio models (for CATNIPS comparisons) are not found (expected in {nerfstudio_dir}). "
              "Skipping setting up nerfstudio")
        
    unnorm_dir = os.path.join(data_dir, "models", "unnormalized_3dgs")
    if os.path.isdir(unnorm_dir):
        baselines_data['splat_nav']['splats']['models_path'] = unnorm_dir
    else:
        print(f"The Un-Normalized 3DGS models (for SplatNav comparisons) are not found (expected in {unnorm_dir}). "
              "Skipping setting up Un-Normalized 3DGS")
    
    with open(baselines_config, 'w') as f:
        yaml.dump(baselines_data, f)
    print(f"Location of SPLANNING data registered successfully as {model_dir} in experiments.yml and in collisions_baselines/base.yml")

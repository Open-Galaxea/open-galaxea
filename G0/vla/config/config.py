import argparse
from omegaconf import OmegaConf
from pathlib import Path
import math

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("round_up", math.ceil)
OmegaConf.register_new_resolver("round_down", math.floor)

def get_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config file")
    args, remaining_args = parser.parse_known_args()
    
    # laod yaml config file
    base_config = OmegaConf.load(args.config)
    OmegaConf.resolve(base_config)
    base_config.config = args.config

    # e.g.: model.lr=0.01 train.batch_size=64
    cli_config = OmegaConf.from_cli(remaining_args)
    
    # merge config file and cli config, cli priority
    merged_config = OmegaConf.merge(base_config, cli_config)
    
    # print final config
    # print(OmegaConf.to_yaml(merged_config))

    return merged_config

def load_cfg(config_path: str = None):
    
    # laod yaml config file
    base_config = OmegaConf.load(config_path)
    OmegaConf.resolve(base_config)

    return base_config
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import sys
import os
import cv2
import json
import hydra
import lightning as L
import rootutils
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import plotly.graph_objects as go

OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("max", lambda x: max(x))
OmegaConf.register_new_resolver("split", lambda s, idx: s.split('/')[int(idx)])

# Add the project root directory to the Python path
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from src.data.galaxea_lerobot_dataset import GalaxeaLerobotDataset
from src.utils.pytorch_utils import dict_apply
from src.utils.visualize import plot_result

def dict_to_array(x):
    data = np.concatenate([item for _, item in x.items()], axis=-1)
    return data

@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)
    
    output_dir = Path(os.path.abspath(os.path.expanduser(cfg.paths.output_dir)))
    output_dir.mkdir(exist_ok=True)
    print(f"Output dir: {output_dir}")
    
    # load model
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    model.load_state_dict(torch.load(cfg.ckpt_path, map_location="cuda:0", weights_only=False)["state_dict"])
    policy: DiffusionUnetImagePolicy = model.policy.cuda().eval()
    
    datamodule.setup('predict')
    dataloader = datamodule.val_dataloader()
    dataset: GalaxeaLerobotDataset = datamodule.data_val

    gt_actions = []
    pd_actions = []
    for i, batch in tqdm(enumerate(dataloader), desc="inferencing", total=len(dataloader)):
        with torch.no_grad():
            cur_pd_action = policy.predict_action(batch).cpu().numpy()
        cur_gt_action = dict_apply(batch["gt_action"], lambda x: x.cpu().numpy())
        gt_actions.append(dict_to_array(cur_gt_action))
        pd_actions.append(cur_pd_action)
    gt_actions = np.concatenate(gt_actions, axis=0)[:, 0, :]
    pd_actions = np.concatenate(pd_actions, axis=0)

    episode_from = dataset.episode_data_index["from"]
    episode_to = dataset.episode_data_index["to"]
    for idx in range(dataset.num_episodes):
        cur_path = output_dir / f"{idx:06}"
        cur_path.mkdir(exist_ok=True)
        cur_gt_action = gt_actions[episode_from[idx]: episode_to[idx]]
        cur_pd_action = pd_actions[episode_from[idx]: episode_to[idx]]
        plot_result(cur_path, cur_gt_action, cur_pd_action)


if __name__ == "__main__":
    main()
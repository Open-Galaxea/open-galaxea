import json
import os
from pathlib import Path
from typing import List, Optional, Union
import packaging

import torch
import numpy as np

from huggingface_hub import HfFileSystem, hf_hub_download

import sys

from paligemma.processing_paligemma import PaliGemmaProcessor
from paligemma.configuration_paligemma import PaliGemmaConfig

from vla.galaxea_zero import GalaxeaZero
from vla.config.import_utils import get_obj_from_str

import logging

logger = logging.getLogger(__name__)

# handling numpy arrays in checkpoints
if packaging.version.parse(torch.__version__) >= packaging.version.parse("2.4.0"):
    torch.serialization.add_safe_globals(
                [np.core.multiarray._reconstruct, np.ndarray, np.dtype, 
                 np.dtypes.UInt32DType, np.dtypes.Float64DType, np.dtypes.BoolDType]
    )

# === HF Hub Repository ===
HF_HUB_REPO = 'google/paligemma2-3b-pt-224'

def load(
    model_cls: GalaxeaZero,
    model_id_or_path: Union[str, Path],
    hf_token: Optional[str] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    load_for_training: bool = False,
    action_expert_only: bool = False,
    **model_kwargs,
) -> GalaxeaZero:
    HF_HUB_REPO = f'google/{model_id_or_path}'
    logger.info(f"Downloading `{(model_id_or_path)} from HF Hub")
    with logger.local_zero_first():
        config_json = hf_hub_download(repo_id=HF_HUB_REPO,token=hf_token, filename="./config.json", cache_dir=cache_dir)

    # Load Model Config from `config.json`
    with open(config_json, "r") as f:
        model_config = json.load(f)
    model_cfg = {
        'model_id': model_config['model_type'],
        'vision_backbone_id': model_config['vision_config']['model_type'],
        'llm_backbone_id': model_config['text_config']['model_type'],
    }
    # = Load Individual Components necessary for Instantiating a VLM =
    #   =>> Print Minimal Config
    logger.info(
        f"Found Config =>> Loading & Freezing [bold blue]{model_cfg['model_id']}[/] with:\n"
        f"             Vision Backbone =>> [bold]{model_cfg['vision_backbone_id']}[/]\n"
        f"             LLM Backbone    =>> [bold]{model_cfg['llm_backbone_id']}[/]\n"
    )

    # Load Vision Backbone
    logger.info(
        f"Loading Vision Backbone [bold]{model_cfg['vision_backbone_id']}[/]\n"
        f"Loading Pretrained LLM [bold]{model_cfg['llm_backbone_id']}[/]\n via HF Transformers")

    # make a fake variable with attribute so that we can bypass the check in the model
    from collections import namedtuple
    model = namedtuple('model', ['vision_tower', 'language_model', 'multi_modal_projector'])(None, None, None)
    model_config = PaliGemmaConfig.from_pretrained('google/paligemma-3b-pt-224', local_files_only=True)
        
    processor = PaliGemmaProcessor.from_pretrained(HF_HUB_REPO, token=hf_token, local_files_only=True)
    
    # Load VLM using `from_pretrained` (clobbers HF syntax... eventually should reconcile)
    logger.info(f"Loading VLA [bold blue]{model_cfg['model_id']}[/] from Checkpoint")
    vla = model_cls.from_pretrained(
        HF_HUB_REPO,
        model_config,
        model.vision_tower,
        model.language_model,
        model.multi_modal_projector,
        tokenizer = processor.tokenizer,
        image_processor = processor.image_processor,
        training=load_for_training,
        action_expert_only=action_expert_only,
        **model_kwargs
    )

    return vla

def load_from_checkpoint(
    checkpoint_pt: Union[str, Path],
    hf_token: Optional[str] = None,
    load_for_training: bool = False,
    action_expert_only: bool = False,
    strict=False,
    **model_kwargs
):      
    # Load Model Config from `config.json` and Model Checkpoint
    logger.info(f"Loading from local checkpoint path `{(checkpoint_pt := Path(checkpoint_pt))}`")
    # [Validate] Checkpoint Path should look like `.../<RUN_ID>/checkpoints/<CHECKPOINT_PATH>.pt`
    assert (checkpoint_pt.suffix == ".pt"), "Invalid checkpoint!"
    run_dir = checkpoint_pt.parents[0]

    # Get paths for `config.json`, `dataset_statistics.json` and pretrained checkpoint
    config_json, dataset_statistics_json = run_dir / "config.json", run_dir / "dataset_statistics.json"

    state_dict = torch.load(checkpoint_pt, weights_only=True, map_location='cpu')

    if "dataset_statistics" in state_dict.keys():
        norm_stats = state_dict.pop("dataset_statistics")
    elif dataset_statistics_json.exists():
        with open(dataset_statistics_json, "r") as f:
            norm_stats = json.load(f)
    else:
        logger.warning(f"No dataset statistics found in")
        norm_stats = {}
    for dataset_name, stats in norm_stats.items():
        if isinstance(stats, dict):
            for k in stats.keys():
                if isinstance(stats[k], dict):
                    for stats_type, v2 in stats[k].items():
                            norm_stats[dataset_name][k][stats_type] = np.array(v2)

    
    if "model" in state_dict:
        state_dict = state_dict["model"]
    elif "module" in state_dict:
        state_dict = state_dict["module"]

    # = Load Individual Components necessary for Instantiating a VLA (via base VLM components) =
    #   =>> Print Minimal Config
    logger.info(
        f"             Checkpoint Path =>> [underline]`{checkpoint_pt}`[/]"
    )
    # Load LLM Backbone --> note `inference_mode = True` by default when calling `load()`
    logger.info(f"Loading [bold]image Processor and tokenizer [/]")
    HF_HUB_REPO = f"google/{model_kwargs['model_cfg']['vla_name']}"
    load_inside = model_kwargs['model_cfg'].get('load_inside', False)

    model_config = PaliGemmaConfig.from_pretrained('google/paligemma-3b-pt-224', local_files_only=True)
        
    processor = PaliGemmaProcessor.from_pretrained(HF_HUB_REPO, token=hf_token, local_files_only=True)

    # Load VLM using `from_pretrained` (clobbers HF syntax... eventually should reconcile)
    logger.info(f"Loading [bold blue]Galaxea VLA[/] from Checkpoint")
    vla_cls: GalaxeaZero = get_obj_from_str(model_kwargs['model_cfg']['name'])
    vla = vla_cls.from_checkpoint(
        HF_HUB_REPO,
        model_config,
        state_dict=state_dict,
        tokenizer = processor.tokenizer,
        image_processor = processor.image_processor,
        norm_stats=norm_stats,
        training=load_for_training,
        action_expert_only=action_expert_only,
        strict=strict,
        **model_kwargs
    )
    
    return vla

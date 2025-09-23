from pathlib import Path
import datetime

import torch
import tyro
import tqdm
import pickle

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.configs.types import FeatureType

from loguru import logger


def main(task: str):
    # Create a directory to store the training checkpoint.
    exp_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_directory = Path(f"outputs/train/{task}/diffusion/{exp_id}")
    output_directory.mkdir(parents=True, exist_ok=True)

    # # Select your device
    device = torch.device("cuda")

    # Number of offline training steps (we'll only do offline training for this example.)
    # Adjust as you prefer. 5000 steps are needed to get something worth evaluating.

    # When starting from scratch (i.e. not from a pretrained policy), we need to specify 2 things before
    # creating the policy:
    #   - input/output shapes: to properly size the policy
    #   - dataset stats: for normalization and denormalization of input/outputs
    dataset_name = task
    dataset_metadata = LeRobotDatasetMetadata(f"galaxea/{dataset_name}")
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    # Policies are initialized with a configuration class, in this case `DiffusionConfig`. For this example,
    # we'll just use the defaults and so no arguments other than input/output features need to be passed.
    cfg = DiffusionConfig(
        input_features=input_features, 
        output_features=output_features,
        crop_shape=(224, 224),
        crop_is_random=False,
        use_separate_rgb_encoder_per_camera=True,
        optimizer_lr=1e-4,
        n_obs_steps=1,
        drop_n_last_frames=8,
    )

    # We can now instantiate our policy with this config and the dataset stats.
    policy = DiffusionPolicy(cfg, dataset_stats=dataset_metadata.stats)
    policy.train()
    policy.to(device)

    # Another policy-dataset interaction is with the delta_timestamps. Each policy expects a given number frames
    # which can differ for inputs, outputs and rewards (if there are some).
    delta_timestamps = {
        "observation.images.rgb_head": [0.0],
        "observation.images.rgb_left_hand": [0.0],
        "observation.images.rgb_right_hand": [0.0],
        "observation.state": [i / dataset_metadata.fps for i in cfg.observation_delta_indices],
        "action": [i / dataset_metadata.fps for i in cfg.action_delta_indices],
    }

    # We can then instantiate the dataset with these delta_timestamps configuration.
    dataset = LeRobotDataset(f"galaxea/{dataset_name}", delta_timestamps=delta_timestamps)
    num_epochs = 300
    batch_size = 128
    training_steps = num_epochs * len(dataset) // batch_size
    log_freq = 50
    save_freq = 1000
    
    print(f"Training for {training_steps} steps.")
    print(f"Logging every {log_freq} steps.")

    # Then we create our optimizer and dataloader for offline training.
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=4,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )

    # Run training loop.
    step = 0
    done = False
    # pbar = tqdm.tqdm(total=training_steps)
    while not done:
        for batch in dataloader:
            batch.pop("task", None)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            loss, _ = policy.forward(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # pbar.update(1)

            if step % log_freq == 0:
                # print(f"step: {step} loss: {loss.item():.3f}")
                # pbar.set_description(f"step: {step} loss: {loss.item():.3f}")
                logger.info(
                    f"step: {step} loss: {loss.item():.3f} lr: {optimizer.param_groups[0]['lr']:.6f} progress: {step / training_steps * 100:.2f}%"
                )
            if step % save_freq == 0:
                # Save a policy checkpoint.
                output_directory.mkdir(parents=True, exist_ok=True)  
                policy.save_pretrained(output_directory / f"checkpoint-{step}")
                with open(output_directory / f"checkpoint-{step}/dataset_metadata.pkl", "wb") as f:
                    pickle.dump(dataset_metadata, f)
            step += 1
            if step >= training_steps:
                done = True
                break

    # Save a policy checkpoint.
    policy.save_pretrained(output_directory / "checkpoint-final")


if __name__ == "__main__":
    tyro.cli(main)
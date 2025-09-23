"""
Reference:
- https://github.com/real-stanford/diffusion_policy
"""
from typing import Dict, Union

import torch
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from einops import reduce
from src.utils.diffusion_policy import LinearNormalizer

from .base_image_policy import BaseImagePolicy
from ..diffusion.conditional_unet1d import ConditionalUnet1D
from .signal_transform import SignalTransform


class DiffusionUnetImagePolicy(BaseImagePolicy):
    def __init__(
        self,
        shape_meta, 
        action_keys, 
        qpos_keys, 
        train_transforms: dict,
        eval_transforms: dict,
        noise_scheduler: Union[DDPMScheduler],
        horizon,
        n_vision_obs_steps,
        n_qpos_obs_steps, 
        obs_encoder,
        rotation_type="quaternion", 
        use_relative_control=False, 
        num_inference_steps=None,
        diffusion_step_embed_dim=256,
        down_dims=(256, 512, 1024),
        kernel_size=5,
        n_groups=8,
        condition_type="film",
        use_down_condition=True,
        use_mid_condition=True,
        use_up_condition=True,
        # parameters passed to step
        **kwargs,
    ):
        super().__init__()
        self.obs_shape_meta = shape_meta["obs"]
        self.condition_type = condition_type
        self.train_transforms = train_transforms
        self.eval_transforms = eval_transforms
        self.current_epoch = None
            
        self.signal_transform = SignalTransform(
            action_keys, 
            qpos_keys, 
            rotation_type, 
            use_relative_control, 
        )

        action_dim = shape_meta['action']['shape'][0]

        # create diffusion model
        obs_feature_dim = obs_encoder.output_shape()
        global_cond_dim = obs_feature_dim

        model = ConditionalUnet1D(
            input_dim=action_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            condition_type=condition_type,
            use_down_condition=use_down_condition,
            use_mid_condition=use_mid_condition,
            use_up_condition=use_up_condition,
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_vision_obs_steps = n_vision_obs_steps
        self.n_qpos_obs_steps = n_qpos_obs_steps
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

    # ========= inference  ============
    def conditional_sample(
        self,
        trajectory,
        global_cond,
    ):
        # set step values
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        with torch.no_grad():
            for t in self.noise_scheduler.timesteps:
                model_output = self.model(
                    sample=trajectory, timestep=t, local_cond=None, global_cond=global_cond
                )
                # compute previous image: x_t -> x_t-1
                trajectory = self.noise_scheduler.step(
                    model_output, t, trajectory, 
                ).prev_sample
        return trajectory

    def predict_action(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        input batch["obs"], output batch["action"]
        """
        batch = self.signal_transform.forward(batch)
        nobs = self.normalizer.normalize(batch["obs"])
        
        value = next(iter(nobs.values()))
        batch_size = value.shape[0]

        if self.eval_transforms is not None:
            for key in self.eval_transforms:
                assert key in nobs
                nobs[key] = nobs[key].flatten(0, 1)
                for trans in self.eval_transforms[key]:
                    nobs[key] = trans(nobs[key])
                nobs[key] = nobs[key].unflatten(0, (batch_size, self.n_vision_obs_steps))
                assert nobs[key].shape[-2:] == self.obs_shape_meta[key]["transform_shape"]

        for key in nobs:
            if "rgb" in key:
                nobs[key] = nobs[key].to(torch.float32) / 255.0

        nobs_features = self.obs_encoder(nobs)
        global_cond = nobs_features.reshape(batch_size, -1)
        
        # empty data for action
        trajectory = torch.randn(
            size=(batch_size, self.horizon, self.action_dim), 
            device=global_cond.device, 
            dtype=global_cond.dtype
        )

        # run sampling
        nsample = self.conditional_sample(
            trajectory,
            global_cond=global_cond,
        )
        # unnormalize prediction
        naction_pred = nsample[..., :self.action_dim]
        action_pred = self.normalizer["action"].unnormalize(naction_pred)
        # get action
        start = self.n_vision_obs_steps - 1
        batch["action"] = action_pred[:, start:]
        batch = self.signal_transform.backward(batch)
        batch["action"] = torch.cat([batch["action"][key] for key in batch["action"]], dim=-1)
        return batch["action"]

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        batch = self.signal_transform.forward(batch)

        # normalize input
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        batch_size = nactions.shape[0]
        
        if self.train_transforms is not None:
            for key in self.train_transforms:
                assert key in nobs
                nobs[key] = nobs[key].flatten(0, 1)
                for trans in self.train_transforms[key]:
                    nobs[key] = trans(nobs[key])
                nobs[key] = nobs[key].unflatten(0, (batch_size, self.n_vision_obs_steps))
                assert nobs[key].shape[-2:] == self.obs_shape_meta[key]["transform_shape"]
        
        for key in nobs:
            if "rgb" in key:
                nobs[key] = nobs[key].to(torch.float32) / 255.0

        nobs_features = self.obs_encoder(nobs)
        # reshape back to B, Do
        global_cond = nobs_features.reshape(batch_size, -1)

        # Sample noise that we'll add to the images
        trajectory = nactions
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps
        )

        # Predict the noise residual
        pred = self.model(
            sample=noisy_trajectory.float(), 
            timestep=timesteps, 
            local_cond=None, 
            global_cond=global_cond
        )

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        elif pred_type == 'v_prediction':
            # https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
            # https://github.com/huggingface/diffusers/blob/v0.11.1-patch/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
            # sigma = self.noise_scheduler.sigmas[timesteps]
            # alpha_t, sigma_t = self.noise_scheduler._sigma_to_alpha_sigma_t(sigma)
            self.noise_scheduler.alpha_t = self.noise_scheduler.alpha_t.to(self.device)
            self.noise_scheduler.sigma_t = self.noise_scheduler.sigma_t.to(self.device)
            alpha_t, sigma_t = self.noise_scheduler.alpha_t[timesteps], self.noise_scheduler.sigma_t[timesteps]
            alpha_t = alpha_t.unsqueeze(-1).unsqueeze(-1)
            sigma_t = sigma_t.unsqueeze(-1).unsqueeze(-1)
            v_t = alpha_t * noise - sigma_t * trajectory
            target = v_t
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss_log = dict()

        dim_loss_keys = [f"train_diffuse_loss/dim_{i:02d}" for i in range(loss.shape[2])]
        dim_loss_vals = [i for i in loss.detach().mean(dim=(0, 1))]
        loss_log.update(dict(zip(dim_loss_keys, dim_loss_vals)))
        
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        loss_log["train/diffuse_loss"] = loss.detach()
        
        return loss, loss_log

    def get_optimizer(self, cfg) -> torch.optim.Optimizer:
        other_params, pretrained_obs_encoder_params = [], []
        for name, param in self.named_parameters():
            if param.requires_grad:
                if name.startswith("obs_encoder.vision_encoders"):
                    pretrained_obs_encoder_params.append(param)
                else:
                    other_params.append(param)
        
        param_groups = [
            {
                "params": other_params, 
                "weight_decay": cfg.weight_decay, 
                "name": "diffusion_model", 
            },
            {
                "params": pretrained_obs_encoder_params, 
                "weight_decay": cfg.weight_decay, 
                "lr_scale": cfg.pretrained_obs_encoder_lr_scale, 
                "name": "pretrained_obs_encoder", 
            }
        ]
        optimizer = torch.optim.AdamW(
            params=param_groups, 
            lr=cfg.lr, 
            betas=cfg.betas
        )
        return optimizer
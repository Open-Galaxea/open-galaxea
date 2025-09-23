import torch
import torch.nn as nn

from src.utils.rotation_conversions import PoseRotationTransformer
from src.utils.relative_pose import RelativePoseTransformer


class SignalTransform(nn.Module):
    def __init__(
        self, 
        action_keys, 
        qpos_keys, 
        rotation_type, 
        use_relative_control, 
    ):
        super().__init__()
        self.pose_rotation_transformer = PoseRotationTransformer(rotation_type)
        self.relative_pose_transformer = RelativePoseTransformer()
        self.use_relative_control = use_relative_control
        self.action_dims = {}
        for key in action_keys:
            if "pose" in key:
                self.action_dims[key] = self.pose_rotation_transformer.pose_dim
            elif "gripper" in key:
                self.action_dims[key] = 1
            elif "joint" in key or "arm" in key:
                self.action_dims[key] = 6 # NOTE: R1 Lite joint action has 6 dof, while R1 Pro has 7, but not implemented yet
            else:
                raise NotImplementedError
        self.qpos_keys = qpos_keys

    def forward(self, batch):
        for category in ["action", "prev_action"]:
            if category in batch:
                actions = []
                for key in self.action_dims:
                    cur_action = batch[category][key]
                    if "ee_pose" in key:
                        if self.use_relative_control:
                            # get cur action rel to cur qpos
                            obs_key = key.replace("action", "obs") # eg: ee_pose_obs_left
                            base_pose = batch["obs"][obs_key][:, -1:, :] # (B, 1, 7)
                            cur_action = self.relative_pose_transformer.forward(cur_action, base_pose)
                        cur_action = self.pose_rotation_transformer.forward(cur_action)
                    actions.append(cur_action)
                # order: ee_left, gripper_left, ee_right, gripper_right
                batch[category] = torch.cat(actions, dim=-1)
        
        qposes = []
        for key in self.qpos_keys:
            cur_qpos = batch["obs"][key]
            if "ee_pose" in key:
                if self.use_relative_control:
                    # get cur qpos rel to start qpos 
                    obs_key = f"episode_start_{key}" # eg: episode_start_ee_pose_obs_left
                    base_pose = batch["obs"][obs_key]
                    if self.training:
                        base_pose = self.pose_rotation_transformer.add_noise(base_pose)
                    ee_pose_wrt_episode_start = self.relative_pose_transformer.forward(cur_qpos, base_pose)
                    ee_pose_wrt_episode_start = self.pose_rotation_transformer.forward(ee_pose_wrt_episode_start)
                    qposes.append(ee_pose_wrt_episode_start)

                    # get cur qpos rel to cur qpos
                    base_pose = cur_qpos[:, -1:, :] # (B, 1, 7)
                    cur_qpos = self.relative_pose_transformer.forward(cur_qpos, base_pose)
                cur_qpos = self.pose_rotation_transformer.forward(cur_qpos)
            qposes.append(cur_qpos)
         # order: ee_left, ee_left_wrt_start, gripper_left, ee_right, ee_right_wrt_start, gripper_right
        batch["obs"]["qpos"] = torch.cat(qposes, dim=-1)

        return batch

    def backward(self, batch):
        actions = {}
        idx = 0
        for key, dim in self.action_dims.items():
            cur_action = batch["action"][:, :, idx: idx + dim]
            idx += dim
            if "ee_pose" in key:
                cur_action = self.pose_rotation_transformer.backward(cur_action)
                if self.use_relative_control:
                    obs_key = key.replace("action", "obs")
                    base_pose = batch["obs"][obs_key][:, -1:, :]
                    # HACK: in policy.prediction, nobs has assigned device, not batch["obs"]
                    base_pose = base_pose.to(cur_action.device)
                    cur_action = self.relative_pose_transformer.backward(cur_action, base_pose)
                actions[key] = cur_action
            else:
                actions[key] = cur_action
        batch["action"] = actions
        return batch

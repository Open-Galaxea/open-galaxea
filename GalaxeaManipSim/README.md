<!-- TITLE & BADGES -->
# 🤖 Galaxea Manipulation Simulator

<p align="center">
  <img src="assets/webp/R1LiteShoePlace/solution.webp" alt="R1Lite" width="30%">
  <img src="assets/webp/R1ProBlocksStackHard/solution.webp" alt="R1" width="30%">
  <img src="assets/webp/R1MugHangingHard/solution.webp" alt="R1Pro" width="30%">
</p>

<p align="center">
  <!-- Python version -->
  <img src="https://img.shields.io/badge/python-3.10+-yellow.svg" alt="Python">
  <!-- CUDA -->
  <img src="https://img.shields.io/badge/cuda-11.8-green.svg" alt="CUDA 11.8">
  <!-- License -->
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License">
  </a>
</p>

Galaxea Manipulation Simulator provides **simulation benchmarks**, **expert demonstration pipelines**, and **baseline policies** for the **Galaxea R1 series** (R1, R1 Pro, R1 Lite).  
Use it to **collect high-quality demos**, **convert them to [LeRobot](https://github.com/huggingface/lerobot) datasets**, and **train / evaluate Diffusion Policies** with minimal overhead.


## ✨ Features
| &nbsp; | &nbsp; |
| :--- | --- |
| 🌍 **Diverse Environments** | **30+** feasible environments spanning **17** distinct tasks. |
| 🛠️ **Turn-key benchmark** | *One-command* setup—downloads assets & registers all simulation tasks automatically. |
| 🎮 **Multi-controller support** | Ready-to-use **joint-space** and **relaxed-ik (EEF-space)** controllers. |
| 📦 **LeRobot compatible** | Seamless **demo → dataset** conversion for downstream training. |
| 🚀 **Baseline DP training** | Drop-in scripts to train / evaluate **Diffusion Policy** models. |
| 📊 **Metrics & videos** | Built-in evaluation with success-rate logging and optional video export. |

<p align="center">
  <img src="assets/mosaic.webp" alt="Environment Teaser" width="100%">
</p>

## 🚀 Installation

**Prerequisites:** Linux + CUDA GPU

Note: If installing together with Galaxea-DP, please refer to this [link]()
```
# Create conda environment
conda create -n galaxea-sim python=3.10 -y
conda activate galaxea-sim
pip install -e .

# Install lerobot
cd ..
git clone https://github.com/huggingface/lerobot.git
cd lerobot
git checkout a5e0aae13a3efd0080ac7ab6b461980d644014ab
pip install -e .
export PYTHONPATH="your_lerobot_codebase_path:$PYTHONPATH"
```

**Heads-up:** If you find `dataset` package version conflict. Please clear the package cache in `~/.cache/huggingface/datasets`.

**Download and extract assets:**
```
gdown https://drive.google.com/file/d/1ZvtCv1H4FLrse_ElUWzsVDt8xRK4CyaC/
unzip robotwin_models.zip
mv robotwin_models galaxea_sim/assets/
```


## 🎮 Collect Demos
### Supported Robots, Tasks, and Controllers

| Robots   | Example Tasks (`Env-Name`)     |  Supported Controllers      |
|---------|------------------------------|-------------------------------------------------------------|
| **R1**      | `R1DualBottlesPickEasy`   | `bimanual_joint_position` / `bimanual_relaxed_ik`           |
| **R1 Pro**  | `R1ProBlocksStackEasy`    | `bimanual_joint_position` / `bimanual_relaxed_ik`           |
| **R1 Lite** | `R1LiteBlocksStackEasy`   | `bimanual_joint_position`         |


### 1. Generate Raw Demos by mplib

```
# Example of R1 Picking up Bottles
python -m galaxea_sim.scripts.collect_demos --env-name R1DualBottlesPickEasy --num-demos 100

# Example of R1 Pro Stacking Blocks
python -m galaxea_sim.scripts.collect_demos --env-name R1ProBlocksStackEasy --num-demos 100

# Example of R1 Lite Stacking Blocks:
python -m galaxea_sim.scripts.collect_demos --env-name R1LiteBlocksStackEasy --num-demos 100
```

Note: The default `--obs_mode` is `state`, which is faster so it is recommended for eef policies that will get image observations during replay. By default, the data will be stored in `datasets/<env-name>/<data-time>` as h5 files.

### 2. Replay Demos by Assigning Controllers


```
# Example of R1 Picking up Bottles by Joints Control
python -m galaxea_sim.scripts.replay_demos --env-name R1DualBottlesPickEasy --target_controller_type bimanual_joint_position --num-demos 100

# Example of R1 Pro Stacking Blocks by End Effector Control
python -m galaxea_sim.scripts.replay_demos --env-name R1ProBlocksStackEasy --target_controller_type bimanual_relaxed_ik --num-demos 100

# Example of R1 Lite Stacking Blocks by Joint Control
python -m galaxea_sim.scripts.replay_demos --env-name R1LiteBlocksStackEasy  --target_controller_type bimanual_joint_position --num-demos 100
```
Note: Replay demonstrations will pass the recorded end effector's pose trajectory to the relaxed_ik controller and check if task is completed. The demos with inverse kinematics solutions will be filtered out, and image/depth observations will be saved. The output will be stored in `datasets/<env-name>/final`.
## 🛠  Train Policies

### 1. Convert Demos to LeRobot Dataset
If you want to use Galaxea Diffusion Policy implementation, just replace the script name as `convert_single_galaxea_sim_to_lerobot_opendp`. If the controller is bimanual_relaxed_ik, please add `--use_eef`.

```
# If you want to use Lerobot Diffusion Policy implementation, please use script convert_single_galaxea_sim_to_lerobot
# For Joint Control
# Example of R1 Picking up Bottles 
python -m galaxea_sim.scripts.convert_single_galaxea_sim_to_lerobot --task R1DualBottlesPickEasy --tag final --robot r1

# Example of R1 Pro Stacking Blocks
python -m galaxea_sim.scripts.convert_single_galaxea_sim_to_lerobot --task R1ProBlocksStackEasy --tag final --robot r1_pro

# Example of R1 Lite Stacking Blocks
python -m galaxea_sim.scripts.convert_single_galaxea_sim_to_lerobot --task R1LiteBlocksStackEasy --tag final --robot r1_lite


# For End Effector Control
# Example of R1 Pro Picking up Bottles 
python -m galaxea_sim.scripts.convert_single_galaxea_sim_to_lerobot --robot r1_pro --task R1ProDualBottlesPickEasy --tag final --use_eef

# Example of R1 Pro Stacking Blocks 
python -m galaxea_sim.scripts.convert_single_galaxea_sim_to_lerobot --robot r1_pro --task R1ProBlocksStackEasy --tag final --use_eef
```

Note: An optional argument `--use_video` makes lerobot storage image data by encoding them into video, which has smaller file size and can be directly played. It requires ffmpeg installed, and you may want to change vcodec to "libh264" in `.../site-packages/lerobot/common/datasets/video_utils.py` if default "libsvtav1" is not available.

`--tag` determines which demos will be convert, by defult it will convert all demos in `datasets/<env-name>`. If `--use_eef` is used, the arm's observation and action will be replace by end effector's pose(x,y,z,qw,qx,qy,qz). The lerobot dataset will be generated in HF_LEROBOT_HOME, it is `~/.cache/huggingface/lerobot` if not specified.


### 2. Data Structure after Converting
#### 2.1 Galaxea DP
```
# Images and depths
observation.images.head_rgb: (224, 224, 3)
observation.images.left_wrist_rgb: (224, 224, 3)
observation.images.right_wrist_rgb: (224, 224, 3)
observation.depth.head_depth： （224，224）

# States and actions
# For eef controller
# arm_dof is 6 when using R1 and R1 Lite, is 7 when using R1 Pro
observation.state.left_arm_joints:  (arm_dof,)
observation.state.left_gripper:     (1,)
observation.state.right_arm_joints: (arm_dof,)
observation.state.right_gripper:    (1,)
action.left_arm_joints:             (arm_dof,)
action.left_gripper:                (1,)
action.right_arm_joints:            (arm_dof,)
action.right_gripper:               (1,)

# States and actions
# For joints controller
observation.state.left_ee_pose:  (7,)
observation.state.left_gripper:  (1,)
observation.state.right_ee_pose: (7,)
observation.state.right_gripper: (1,)
action.left_ee_pose:             (7,)
action.left_gripper:             (1,)
action.right_ee_pose:            (7,)
action.right_gripper:            (1,)
```
#### 2.2 Lerobot DP
```
# Images and depths
observation.images.rgb_head: (224, 224, 3)
observation.images.rgb_left_hand: (224, 224, 3)
observation.images.rgb_right_hand: (224, 224, 3)

# States and actions
# For eef controller
# arm_dof is 6 when using R1 and R1 Lite, is 7 when using R1 Pro
observation.state:  (2*arm_dof + 2,)
action:             (2*arm_dof + 2,)

# States and actions
# For joints controller
observation.state:  (16,)
action:             (16,)
```

### 3. Train Diffusion Policy
#### 3.1 Lerobot Diffusion Policy
By default, the policy will be saved in `outputs/train/<env-name>/diffusion/<date-time>`. Restoring from checkpoints is currently not supported.

```
# Example of R1 Picking up Bottles 
python -m galaxea_sim.scripts.train_lerobot_dp_policy --task R1DualBottlesPickEasy

# Example of R1 Pro Stacking Blocks 
python -m galaxea_sim.scripts.train_lerobot_dp_policy --task R1ProBlocksStackEasy

# Example of R1 Lite Stacking Blocks 
python -m galaxea_sim.scripts.train_lerobot_dp_policy --task R1LiteBlocksStackEasy
```
#### 3.2 Galaxea Diffusion Policy
Please refer to Galaxea DP repository in this [link]().
### 4. Evaluate LeRobot Diffusion Policy
#### 4.1 Lerobot Diffusion Policy
The evaluation result will be saved in the specified checkpoint dir; use `--save-video` to save videos of the evaluation.
```
# Example of R1 Picking up Bottles 
python -m galaxea_sim.scripts.eval_lerobot_dp_policy --task R1DualBottlesPickEasy --pretrained-policy-path outputs/train/R1DualBottlesPickEasy/diffusion/.../checkpoint --target_controller_type bimanual_joint_position 

# Example of R1 Pro Stacking Blocks
python -m galaxea_sim.scripts.eval_lerobot_dp_policy --task R1ProBlocksStackEasy --pretrained-policy-path outputs/train/R1ProBlocksStackEasy/diffusion/.../checkpoint  --target_controller_type bimanual_joint_position

# Example of R1 Lite Stacking Blocks 
python -m galaxea_sim.scripts.eval_lerobot_dp_policy --task R1LiteBlocksStackEasy --pretrained-policy-path outputs/train/R1LiteBlocksStackEasy/diffusion/.../checkpoint --target_controller_type bimanual_joint_position
```
#### 3.2 Galaxea Diffusion Policy
Please refer to Galaxea DP repository in this [link]().
## 📈 Success Rate

|       Task              |   Robot   | OpenDP Success Rate | LeRobot Success Rate |
|:-----------------------:|:---------:|:-------------------:|:--------------------:|
| Dual Bottles Pick Easy |    R1     |        98%        |        98%         |
| Dual Bottles Pick Easy |  R1 Pro   |        98%        |        98%         |
| Blocks Stack Easy      |  R1 Pro   |        68%        |        64%         |
| Blocks Stack Easy      |  R1 Lite  |        51%        |        42%         |

> Note: These results are based on 100 evaluation rollouts. "OpenDP" refers to the diffusion policy implemented by Galaxea. "LeRobot" refers to the policy implemented by LeRobot. For brevity, all success rates are evaluated with the joints controller.


## 🏷️ License
This repository is released under the MIT license. See [LICENSE](LICENSE) for additional details.


## 🙏 Acknowledgements
Our code is generally built on top of amazing open-source projects:
[Diffusion Policy](https://github.com/real-stanford/diffusion_policy), [LeRobot](https://github.com/huggingface/lerobot), [Robotwin](https://github.com/robotwin-Platform/RoboTwin).



## 📚 Citation

If you find our work useful, please consider citing:
```
@inproceedings{GalaxeaManipSim,
  title={Galaxea Manipulation Simulator},
  author={Galaxea Team},
  year={2025}
}
```
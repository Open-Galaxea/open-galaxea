# Galaxea R1LITE RealWorld Inference

## Setup Real Robot

```
ssh r1lite@<robot-ip>

export ROS_MASTER_URI=http://<robot-ip>
export ROS_IP=<robot-ip>
cd <r1lite-sdk>/install/share/start_configs/scripts
./robot_startup.sh boot ../session.d/ATCStandard/R1LITEBody.d/
```

## Check Interface on Workstation

Guarantee your GPU memory meets the requirements.

**Setup environment variables:**

```
export ROS_MASTER_URI=http://<robot-ip>
export ROS_IP=<robot-ip>
```

**Setting Language Instructions**

Set your language instruction [here](experiments/robot/galaxea_real/instruction.txt). At each inference step, language instruction file will be read into model.

(The GUI language instructions switcher code comes soon.)

**Running Inference**

```
cd G0

python experiments/robot/galaxea_real/run.py --interface-config.with_chassis --interface-config.with_torso --run_dir runs/<your-folder-name> --ckpt-id 12500 --dtype fp32
```

**Replay ROS Bag (Optional)**

For code debugging, it is more convenient to replay a pre-collected ROS Bag and running inference on this.

You can also remap the teleoperation action topics to the ground truth topics and compare inference actions with the ground truth.

```
rosbag play <YOUR ORIGIN ROS BAG PATH HERE> \
/motion_target/target_joint_state_arm_left:=/motion_target/target_joint_state_arm_left_gt \
/motion_target/target_joint_state_arm_right:=/motion_target/target_joint_state_arm_right_gt \
/motion_target/target_position_gripper_left:=/motion_target/target_position_gripper_left_gt \
/motion_target/target_position_gripper_right:=/motion_target/target_position_gripper_right_gt --loop
```


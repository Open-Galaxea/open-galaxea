env=R1ProBlocksStackEasy
task=R1ProBlocksStackEasy_joints
dataset_dir="./data"
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export GALAXEA_DP_WORK_DIR="./out"
target_controller_type=bimanual_relaxed_ik
batch_size=64   # About 10GB GPU memory for 64 batch size
set -e

# collect episodes using mplib, which are saved to $dataset_dir/$env/collected
echo "------------ Collect Demos ------------"
python -m galaxea_sim.scripts.collect_demos \
    --env-name $env \
    --num-demos 125 \
    --dataset_dir $dataset_dir

# replay episodes using IK controller, which are save to $dataset_dir/$env/replayed
echo "------------ Replay Demos ------------"
python -m galaxea_sim.scripts.replay_demos \
    --env-name $env \
    --target_controller_type \
    $target_controller_type \
    --num-demos 100 \
    --dataset_dir $dataset_dir

# convert replayed episodes into GalaxeaLeRobot format, which are saved to $dataset_dir/$env/lerobot
echo "------------ Convert Demos ------------"
python -m galaxea_sim.scripts.convert_single_galaxea_sim_to_galaxea_lerobot \
    --task $env \
    --tag replayed \
    --robot r1_pro \
    --use_eef \
    --dataset_dir $dataset_dir

# train policy, whose data are saved to $GALAXEA_DP_WORK_DIR/sim/$task
echo "------------ Train the DP ------------"
bash scripts/train.sh \
    task=sim/$task \
    data.train.dataset_dirs=[$dataset_dir/$env/lerobot] \
    data.batch_size_train=$batch_size \
    data.batch_size_val=$batch_size

# open loop eval
echo "------------ Open Loop Eval ------------"
latest=$(ls -1 out/sim/$task | sort | tail -n 1)
# latest=2025-09-18_00-45-07
bash scripts/eval_open_loop.sh \
    task=sim/$task \
    ckpt_path=out/sim/$task/$latest/checkpoints/step_05000.ckpt \
    data.train.dataset_dirs=[$dataset_dir/$env/lerobot] \

# rollout eval in simulation
echo "------------ Rollout Eval in Sim ------------"
eval_output=$(ls -1 out/sim/$task | sort | tail -n 1)
bash scripts/eval_sim.sh \
    task=sim/$task \
    ckpt_path=out/sim/$task/$latest/checkpoints/step_05000.ckpt \
    env=$env \
    target_controller_type=$target_controller_type

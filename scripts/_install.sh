#!/bin/bash

# -------------------------------
# Open-Galaxea 一键安装脚本
# -------------------------------

# 1️⃣ 创建 conda 环境（如果不存在）
# ENV_NAME="opengalaxea"
# PYTHON_VER="3.10"

# if conda env list | grep -q "$ENV_NAME"; then
#     echo "Conda 环境 '$ENV_NAME' 已存在，跳过创建..."
# else
#     echo "创建 conda 环境 '$ENV_NAME'..."
#     conda create -n $ENV_NAME python=$PYTHON_VER -y
# fi

# # 2️⃣ 激活环境
# echo "激活 conda 环境 '$ENV_NAME'..."
# source $(conda info --base)/etc/profile.d/conda.sh
# conda activate $ENV_NAME

# 3️⃣ 安装 Git LFS 并初始化
if ! command -v git-lfs &> /dev/null; then
    echo "安装 Git LFS..."
    sudo apt update && sudo apt install git-lfs -y
fi
git lfs install

# # 4️⃣ 克隆仓库
# REPO_URL="https://github.com/Open-Galaxea/open-galaxea.git"
# if [ -d "open-galaxea" ]; then
#     echo "仓库已存在，跳过克隆..."
# else
#     echo "克隆仓库..."
#     git clone $REPO_URL
# fi

# cd open-galaxea || exit

# 5️⃣ 安装子模块依赖

echo "安装 GalaxeaDP 依赖..."
pip install -r GalaxeaDP/requirements.txt

echo "安装 GalaxeaManipSim 可编辑模式..."
pip install --no-deps -e GalaxeaManipSim

echo "安装 GalaxeaLeRobot 可编辑模式..."
pip install --no-deps -e GalaxeaLeRobot

echo "安装完成！你现在可以使用 Open-Galaxea 项目。"

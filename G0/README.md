# Galaxea Open-World Dataset & G0 Dual-System VLA Model

[![Project Page](https://img.shields.io/badge/Project%20Page-000000?style=for-the-badge&logo=github)](https://opengalaxea.github.io/G0/)
[![Paper](https://img.shields.io/badge/Paper-8A2BE2?style=for-the-badge&logo=arxiv)](https://arxiv.org/abs/2509.00576v1)
[![Videos](https://img.shields.io/badge/Videos-FF0000?style=for-the-badge&logo=youtube)](https://opengalaxea.github.io/G0/)
[![Visualizer](https://img.shields.io/badge/Visualizer-FF8C00?style=for-the-badge&logo=airplayvideo)](https://opengalaxea.github.io/G0/visualizer/index.html)
[![Huggingface](https://img.shields.io/badge/Huggingface-FF6B35?style=for-the-badge&logo=huggingface)](https://huggingface.co/datasets/OpenGalaxea/Galaxea-Open-World-Dataset)
[![Modelscope](https://img.shields.io/badge/Modelscope-1890FF?style=for-the-badge&logo=alibabacloud)](https://www.modelscope.cn/datasets/Galaxea/Galaxea-Open-World-Dataset)
[![Twitter](https://img.shields.io/badge/Twitter-FF6B35?style=for-the-badge&logo=x)](https://x.com/Galaxea_x)
[![Linkedin](https://img.shields.io/badge/Linkedin-1890FF?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/company/galaxeadynamics/posts/?feedView=all&viewAsMember=true)




## ⏰ Roadmap / Release Timeline

We are gradually open-sourcing the dataset and model. Progress will be updated here:

- [x] **Aug 23, 2025**  
  - Release **Galaxea Open-World Dataset**.
  - Now our Open-Galaxea-Dataset is available at [Huggingface](https://huggingface.co/datasets/OpenGalaxea/Galaxea-Open-World-Dataset) and [Modelscope](https://www.modelscope.cn/datasets/Galaxea/Galaxea-Open-World-Dataset)!

- [x] **Sep 9, 2025**  
  - Release **G0-VLA pretrained model weights**.
  - Now our pretained weight is available at [Huggingface](https://huggingface.co/OpenGalaxea/G0-VLA) and [Modelscope](https://www.modelscope.cn/models/Galaxea/G0-VLA)!

- [x] **Sep 17, 2025**  
  - Release **G0-VLA real-robot [inference code](docs/inference.md)**.

- [x] **Sep 17, 2025**  
  - Release **G0-VLA fine-tuning code**.

- [ ] **Mid-Sep, 2025**  
  - Release **Lerobot Format Galaxea Open-World Dataset**.

- [ ] **Later in 2025**  
  - 🔮 More updates to come (extended datasets, improved models, additional tools).


## 📌 Overview

We introduce **Galaxea Open-World Dataset**, a large-scale, high-quality robot behavior dataset collected in **authentic human living and working environments**.  
We also present **G0**, a **dual-system VLA** model that combines:

- **G0-VLM**: a multimodal planner for high-level reasoning and subtask planning.  
- **G0-VLA**: a real-time executor for precise low-level action control.

The dataset and model are designed to **advance real-world, long-horizon, and few-shot robotic manipulation**.

<p align="center">
  <img src="assets/teaser.png" alt="Galaxea Dataset & G0 Dual-System Overview" width="700"/>
</p>



## 🚀 Galaxea Open-World Dataset

### **Key features**

- **500+ hours** of real-world mobile manipulation data.
- All data collected using **one uniform robotic embodiment** for consistency.
- Fine-grained **subtask language annotations**.
- Covers **residential**, **kitchen**, **retail**, and **office** settings.
- Dataset in **RLDS** format.

See more dataset (formats and examples) details [here](docs/dataset.md).

## G0-VLA

#### GPU Requirements

To run our pretrained models in this repository, you will need an NVIDIA GPU with at least the following specifications. These estimations assume a single GPU, but you can also use multiple GPUs with model parallelism to reduce per-GPU memory requirements by configuring `--nnodes` and`--nproc-per-node` in the fine-tune start shell script. 

| Mode               | Memory Required | Example GPU              |
| ------------------ | --------------- | ------------------------ |
| Inference          | > 8 GB          | RTX 3090 / RTX 4090      |
| Fine-Tuning (Full) | > 70 GB         | A100 (80GB) / H20 (96GB) |

#### Installation

```
git clone https://github.com/OpenGalaxea/G0
conda env create -f glx_vla.yml
conda activate g0

# Install Pacakges from Code
git clone https://github.com/kvablack/dlimp
cd dlimp
pip install -e .
```

#### Model Checkpoints

| Model                  | Use Case    | Description                       | Checkpoint Path                                              |
| ---------------------- | ----------- | --------------------------------- | ------------------------------------------------------------ |
| G0_3B_base              | Fine-Tuning | Base G0-VLA Model for fine-tuning | https://huggingface.co/OpenGalaxea/G0-VLA/blob/main/G0_3B_base.pt |
| More Models come soon! |             |                                   |                                                              |

#### Fine-Tuning Base Models on Galaxea R1Lite Robot

To fine-tune our model with your own data, you should follow three steps:

1. Convert your data to a RLDS dataset. You can follow data converter open-sourced by  [OpenVLA](https://github.com/moojink/rlds_dataset_builder).

2. Defining training configs and running training:

   **Defining Training Configs**

   You can find a training config template at `vla/config/r1_lite/r1lite_fine_tune_example.yml`

   **Running Training**

   ```
   cd G0
   conda activate g0
   
   # For Single Nodes Post-Training
   torchrun \
       --standalone \
       --nnodes 1 \
       --nproc-per-node <num-gpus> \
       finetune.py --config <your-training-config-path>

3. Running Inference in with real world with Galaxea R1Lite Robot **(ROS1)**

   See detailed commands and launch methods [here](docs/inference.md).


#### Precision

1. Inference: Support either BF16 or FP32. You can change data type by specifying `dtype` parameter while launching inference.
2. Training: Support either BF16 or FP32. You can enable BF16 by setting `enable_bf16: True` in the training config file. Our open-sourced pretrained weight is trained with BF16.

## Troubleshooting

We will collect common issues and their solutions here. If you encounter an issue, please check here first. If you can't find a solution, please file an issue on the repo.

|     Issue     |                          Resolution                          |
| :-----------: | :----------------------------------------------------------: |
| About dataset | Step in our dataset is 15 HZ, and image resolution in RLDS is 224 x 224. But the lerobot format dataset with full resolution (1280 x 720) will come soon. |
|               |                                                              |
|               |                                                              |



## Acknowledgement

This project builds upon prior work from the open-source community. The implementation was inspired by [open-pi-zero](https://github.com/allenzren/open-pi-zero), [OpenVLA](https://github.com/openvla/openvla), [Octo](https://github.com/octo-models/octo), and [Openpi](https://github.com/Physical-Intelligence/openpi), and the experiments make use of datasets including [OXE](https://github.com/google-deepmind/open_x_embodiment), [RDT](https://github.com/thu-ml/RoboticsDiffusionTransformer), [BridgeV2](https://github.com/rail-berkeley/bridge_data_v2), and [DROID](https://github.com/droid-dataset/droid). We sincerely thank the authors of these projects for making their code and data publicly available.


## 📜 Citation

If you use our dataset or models, please cite:

```bibtex
@article{galaxea2025,
  title={Galaxea G0: Open-World Dataset and Dual-System VLA Model},
  author={Galaxea Team},
  journal={arXiv preprint arXiv:2509.00576v1},
  year={2025}
}
```

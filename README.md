# TrajFlow: Multi-modal Motion Prediction via Flow Matching

[![arXiv](https://img.shields.io/badge/arXiv-Paper-red)](https://www.arxiv.org/abs/2506.08541)
[![Project Webpage](https://img.shields.io/badge/Project_Page-Website-green?logo=googlechrome&logoColor=white)](https://traj-flow.github.io/)
[![Blog](https://img.shields.io/badge/Blog-2ea44f)](https://qiyan98.github.io/blog/2025/cogen-motion/)

The official PyTorch implementation of IROS'25 paper named "TrajFlow: Multi-modal Motion Prediction via Flow Matching".

## Overview

![TrajFlow diagram](assets/trajflow_overview.png)
We propose a new flow matching framework to predict multi-modal trajectories on the large-scale Waymo Open Motion Dataset.

## Docker Setup

The recommended way to run this repo is the provided Docker image. It includes:

- `python==3.10`
- `torch==2.2.0`
- `tensorflow==2.12.0`
- `waymo-open-dataset-tf-2-12-0==1.6.4`
- the compiled TrajFlow CUDA extensions

Build the image from the repo root:

```bash
docker build -t trajflow:cu121 .
```

Prepare local writable directories on the host:

```bash
mkdir -p data/waymo output
```

Download the required intention-points file on the host:

```bash
curl -L https://raw.githubusercontent.com/sshaoshuai/MTR/master/data/waymo/cluster_64_center_dict.pkl -o data/waymo/cluster_64_center_dict.pkl
```

Run an interactive shell with GPUs enabled:

```bash
bash scripts/run_docker.sh
```

The launcher mounts:

- `$(pwd)/data/waymo` to `/workspace/TrajFlow/data/waymo`
- `/data2/datasets/Waymo/waymo_motion_sc` to `/workspace/TrajFlow/data/waymo/scenario` as read-only
- `$(pwd)/output` to `/workspace/TrajFlow/output`
- host GPUs `4,5` into the container

You can override the dataset path or image name:

```bash
WAYMO_SCENARIO_DIR=/path/to/waymo_motion_sc IMAGE_NAME=trajflow:cu121 bash scripts/run_docker.sh
```

You can also run a one-off command instead of an interactive shell:

```bash
bash scripts/run_docker.sh python -c "import torch; print(torch.cuda.is_available())"
```

Inside the container, verify the environment:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"
python -c "from waymo_open_dataset.metrics.ops import py_metrics_ops; print('waymo metrics ok')"
```

The host `nvidia-smi` driver can be newer than CUDA 12.1. The image uses a CUDA 12.1 toolkit because it matches the pinned PyTorch runtime and extension build path.

## Waymo Dataset Preparation

**Step 1:** Download Waymo Open Motion Dataset from the [official website](https://waymo.com/open/download/) and organize the raw scenario files as follows:

```bash
├── data
│   ├── waymo
│   │   ├── scenario
│   │   │   ├──training
│   │   │   ├──validation
│   │   │   ├──testing
├── ...
```

**Step 2:** Preprocess the dataset inside the container:

```bash
cd trajflow/datasets/waymo
python data_preprocess.py ../../../data/waymo/scenario/  ../../../data/waymo
```

The processed data will be saved to `data/waymo/` directory as follows:

```bash
├── data
│   ├── waymo
│   │   ├── processed_scenarios_training
│   │   ├── processed_scenarios_validation
│   │   ├── processed_scenarios_testing
│   │   ├── processed_scenarios_validation_interactive
│   │   ├── processed_scenarios_testing_interactive
│   │   ├── processed_scenarios_training_infos.pkl
│   │   ├── processed_scenarios_val_infos.pkl
│   │   ├── processed_scenarios_test_infos.pkl
│   │   ├── processed_scenarios_val_inter_infos.pkl
│   │   ├── processed_scenarios_test_inter_infos.pkl
├── ...
```

We use the clustering result from [MTR](https://github.com/sshaoshuai/MTR) for intention points, which is saved in `data/waymo/cluster_64_center_dict.pkl`.
This file is required before training or evaluation. If it is missing, model construction will stop with an explicit error.

## Training and Evaluation

The shipped Waymo YAMLs now match the paper recipe more closely. The training parser accepts both `--epochs` and `--epoch` if you want to override them manually.

```bash
## setup wandb credentials if you use it
wandb login

## or disable it completely
export WANDB_DISABLED=true

## training
cd runner
bash scripts/dist_train.sh 4 --cfg_file cfgs/waymo/trajflow+100_percent_data.yaml --batch_size 80 --extra_tag trajflow --max_ckpt_save_num 100 --ckpt_save_interval 1

## evaluation
### validation set
python test.py --batch_size 64 --extra_tag=$(hostname) \
--ckpt ${PATH_TO_CKPT} \
--val --full_eval

### testing set (for submission)
python test.py --batch_size 64 --extra_tag=$(hostname) \
--ckpt ${PATH_TO_CKPT} \
--test --full_eval --submit --email ${EMAIL}  --method_nm ${METHOD_NAME}
```

## Acknowledgment and Contact

We would like to thank the [MTR](https://github.com/sshaoshuai/MTR) and [BeTopNet](https://github.com/OpenDriveLab/BeTop) repositories for their open-source codebase.

If you have any questions, please contact [Qi Yan](mailto:qi.yan@ece.ubc.ca).

## Citation

If you find this work useful, please consider citing:

```bibtex
@article{yan2025trajflow,
      title={TrajFlow: Multi-modal Motion Prediction via Flow Matching},
      author={Yan, Qi and Zhang, Brian and Zhang, Yutong and Yang, Daniel and White, Joshua and Chen, Di and Liu, Jiachao and Liu, Langechuan and Zhuang, Binnan and Shi, Shaoshuai and others},
      journal={arXiv preprint arXiv:2506.08541},
      year={2025}
    }
```

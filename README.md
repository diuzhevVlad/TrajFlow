# TrajFlow: Multi-modal Motion Prediction via Flow Matching

[![arXiv](https://img.shields.io/badge/arXiv-Paper-red)](https://www.arxiv.org/abs/2506.08541)
[![Project Webpage](https://img.shields.io/badge/Project_Page-Website-green?logo=googlechrome&logoColor=white)](https://traj-flow.github.io/)
[![Blog](https://img.shields.io/badge/Blog-2ea44f)](https://qiyan98.github.io/blog/2025/cogen-motion/)

The official PyTorch implementation of IROS'25 paper named "TrajFlow: Multi-modal Motion Prediction via Flow Matching".

## Overview

![TrajFlow diagram](assets/trajflow_overview.png)
We propose a new flow matching framework to predict multi-modal trajectories on the large-scale Waymo Open Motion Dataset.

## Install Python Environment

**Step 1:** Create a python environment

```bash
conda create --name trajflow python=3.10 -y
conda activate trajflow 
```

Please note that we use `python=3.10` mainly for compatibility with the `waymo-open-dataset-tf-2-12-0` package, which is required for metrics evaluation.

**Step 2:** Install the required packages

```bash
# install the pinned runtime stack: pytorch + tensorflow + waymo + project deps
pip install -r setup/requirements.txt
```

This file pins a compatible stack around:

- `torch==2.2.0`
- `tensorflow==2.12.0`
- `waymo-open-dataset-tf-2-12-0==1.6.4`

so `pip` does not silently downgrade `torch` or pull a newer incompatible Waymo metrics wheel.

**Step 3:** Compile CUDA code

```bash
conda install git
conda install -c conda-forge ninja

# a recent NVIDIA driver is enough to run the pinned torch wheel
# but compiling TrajFlow custom ops still requires an nvcc toolkit in your shell
# for torch==2.2.0, CUDA 12.1 is the safest match
export TORCH_CUDA_ARCH_LIST="9.0"
python setup/setup_trajflow.py develop

# if you don't have a compatible toolkit installed, you can install CUDA 12.1 locally
bash setup/install_cuda12_local.sh
source setup/use_local_cuda12.sh
python setup/setup_trajflow.py develop
```

`nvidia-smi` showing a newer driver such as `CUDA Version: 13.0` is fine. That is the driver capability, not the toolkit used for compilation.

Finally, run the following for sanity check:

```bash
python -c "import torch; import trajflow; print(torch.__version__, trajflow.__file__, 'pytorch sanity check pass'); "
python -c "from waymo_open_dataset.metrics.ops import py_metrics_ops; print('waymo metrics sanity check pass'); "
```

## Waymo Dataset Preparation

**Step 1:** Download Waymo Open Motion Dataset `v1.3.0` from the [official website](https://waymo.com/open/download/) at `waymo_open_dataset_motion_v_1_3_0/uncompressed/scenario`, and organize the data as follows:

```bash
├── data
│   ├── waymo
│   │   ├── scenario
│   │   │   ├──training
│   │   │   ├──validation
│   │   │   ├──testing
├── ...
```

**Step 2:** Preprocess the dataset:

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

```bash
mkdir -p data/waymo
curl -L https://raw.githubusercontent.com/sshaoshuai/MTR/master/data/waymo/cluster_64_center_dict.pkl -o data/waymo/cluster_64_center_dict.pkl
```

## Training and Evaluation

The shipped Waymo YAMLs now match the paper recipe more closely. The training parser accepts both `--epochs` and `--epoch` if you want to override them manually.

```bash
## setup wandb credentials
wandb login

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

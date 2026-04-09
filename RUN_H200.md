# TrajFlow on 2x H200

This is a server-oriented checklist to reproduce the main `Waymo Open Motion Dataset` baseline from this repo.

Your raw dataset is currently at:

```bash
/data2/datasets/Waymo/waymo_motion_sc/
├── training
├── validation
└── testing
```

That is enough for the main marginal forecasting setup.

It does **not** include the interactive splits:

- `validation_interactive`
- `testing_interactive`

So follow the standard training and validation path below, and skip interactive evaluation unless you later download those splits.

## 1. Environment

```bash
conda create -n trajflow python=3.10 -y
conda activate trajflow

pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
pip install waymo-open-dataset-tf-2-12-0
pip install -r setup/requirements.txt
```

This now installs `tensorboard` too, which the repo uses for local experiment dashboards.

## 2. Build CUDA Extensions

Yes, you do need this step. The repo uses custom CUDA ops for KNN and attention, and training/inference will not work correctly without compiling them.

From the repo root:

```bash
export TORCH_CUDA_ARCH_LIST="9.0"
python setup/setup_trajflow.py develop
```

If that fails because `nvcc` is missing or incompatible, make sure a CUDA 11.8 toolkit is available in your shell first.

Quick checks:

```bash
python -c "import torch; import trajflow; print(torch.__version__, trajflow.__file__)"
python -c "from waymo_open_dataset.metrics.ops import py_metrics_ops; print('waymo metrics ok')"
```

## 3. Point the Repo at Your Waymo Data

This repo expects raw data under `data/waymo/scenario`.

From the repo root:

```bash
mkdir -p data/waymo
ln -s /data2/datasets/Waymo/waymo_motion_sc data/waymo/scenario
```

Check it:

```bash
ls data/waymo/scenario
```

Expected output:

```bash
testing
training
validation
```

## 4. Download the Intention Points File

From the repo root:

```bash
mkdir -p data/waymo
curl -L https://raw.githubusercontent.com/sshaoshuai/MTR/master/data/waymo/cluster_64_center_dict.pkl -o data/waymo/cluster_64_center_dict.pkl
```

Check it:

```bash
ls -lh data/waymo/cluster_64_center_dict.pkl
```

## 5. Preprocess Waymo

From the repo root:

```bash
cd trajflow/datasets/waymo
python data_preprocess.py ../../../data/waymo/scenario/ ../../../data/waymo
cd ../../..
```

Notes:

- Since your current raw data does not include the interactive splits, the script will only produce meaningful standard train/val/test processed data.
- Do not use `--interactive` evaluation unless you later add `validation_interactive` and `testing_interactive` to the raw dataset.

Check that preprocessing worked:

```bash
ls data/waymo
```

You should see at least:

```bash
cluster_64_center_dict.pkl
processed_scenarios_training
processed_scenarios_validation
processed_scenarios_testing
processed_scenarios_training_infos.pkl
processed_scenarios_val_infos.pkl
processed_scenarios_test_infos.pkl
```

## 6. Train the 100% Baseline on 2 GPUs

From the repo root:

```bash
cd runner
CUDA_VISIBLE_DEVICES=0,1 bash scripts/dist_train.sh 2 \
  --cfg_file cfgs/waymo/trajflow+100_percent_data.yaml \
  --batch_size 80 \
  --extra_tag h200_repro \
  --max_ckpt_save_num 100 \
  --ckpt_save_interval 1 \
  --fix_random_seed
```

Notes:

- `--batch_size 80` is the global batch size, so this becomes `40` per GPU.
- The shipped config is already aligned to the paper recipe more closely:
  - `NUM_EPOCHS: 40`
  - `SCHEDULER: linearLR`
- Training outputs will be written under `output/runner/cfgs/waymo/trajflow+100_percent_data/...`

## 7. Find the Best Checkpoint

After training, check:

```bash
find ../output -name best_model.pth
```

Or inspect the run directory under:

```bash
../output/runner/cfgs/waymo/trajflow+100_percent_data/
```

The best checkpoint is typically:

```bash
.../ckpt/best_model.pth
```

## 8. Run Full Validation

Use full validation and evaluate the EMA weights.

From `runner/`:

```bash
CUDA_VISIBLE_DEVICES=0,1 python test.py \
  --batch_size 64 \
  --ckpt /absolute/path/to/best_model.pth \
  --val \
  --full_eval \
  --ema_coef 0.999 \
  --extra_tag h200_repro_val
```

If you want to compare online weights and EMA weights in one pass:

```bash
CUDA_VISIBLE_DEVICES=0,1 python test.py \
  --batch_size 64 \
  --ckpt /absolute/path/to/best_model.pth \
  --val \
  --full_eval \
  --ema_coef all \
  --extra_tag h200_repro_val_all
```

## 9. What to Compare Against

For the main paper result, compare your validation metrics against the standard WOMD marginal forecasting setup, not interactive.

Focus on:

- `minADE`
- `minFDE`
- `MissRate`
- `mAP`
- `soft mAP`

## 10. What Not to Run Yet

Do **not** run these until you actually have the interactive splits:

```bash
python test.py --val --interactive ...
python test.py --test --interactive ...
```

## 11. Common Failure Points

### Missing CUDA compiler during extension build

Symptom:

- `setup/setup_trajflow.py develop` fails during compilation.

Check:

```bash
which nvcc
nvcc --version
```

### Missing intention points file

Symptom:

- model construction fails with `Missing intention points file`.

Fix:

```bash
ls -lh data/waymo/cluster_64_center_dict.pkl
```

### Missing processed data

Symptom:

- dataset loader fails looking for `processed_scenarios_*` files.

Fix:

- rerun preprocessing
- verify the `data/waymo/scenario` symlink points to your raw data

Check:

```bash
readlink -f data/waymo/scenario
ls data/waymo/scenario
```

### WandB login issue

`wandb` is now optional in this repo.

If you cannot use `wandb`, disable it completely and rely on local files:

```bash
export WANDB_DISABLED=true
```

Or equivalently:

```bash
export WANDB_MODE=disabled
```

Training and evaluation logs will still be written locally under the run output directory, including:

- `log_train_*.txt`
- `eval/.../log_eval_*.txt`
- `eval/.../best_eval_record.txt`
- `eval/.../result_denoiser.pkl`

TensorBoard event files are also written under:

- `output/.../tensorboard`
- `output/.../eval/.../tensorboard`

To view them on the server:

```bash
tensorboard --logdir output --port 6006 --bind_all
```

If you want `wandb` but without syncing to the cloud, use offline mode:

```bash
export WANDB_MODE=offline
```

Then continue with training as usual.

If you do want online `wandb`, either export your API key first:

```bash
export WANDB_API_KEY=your_key_here
```

or do:

```bash
wandb login
```

## 12. Minimal Command Sequence

If you want the shortest possible version, from repo root:

```bash
conda create -n trajflow python=3.10 -y
conda activate trajflow
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
pip install waymo-open-dataset-tf-2-12-0
pip install -r setup/requirements.txt
export TORCH_CUDA_ARCH_LIST="9.0"
python setup/setup_trajflow.py develop
mkdir -p data/waymo
ln -s /data2/datasets/Waymo/waymo_motion_sc data/waymo/scenario
curl -L https://raw.githubusercontent.com/sshaoshuai/MTR/master/data/waymo/cluster_64_center_dict.pkl -o data/waymo/cluster_64_center_dict.pkl
cd trajflow/datasets/waymo
python data_preprocess.py ../../../data/waymo/scenario/ ../../../data/waymo
cd ../../..
cd runner
export WANDB_DISABLED=true
CUDA_VISIBLE_DEVICES=0,1 bash scripts/dist_train.sh 2 --cfg_file cfgs/waymo/trajflow+100_percent_data.yaml --batch_size 80 --extra_tag h200_repro --max_ckpt_save_num 100 --ckpt_save_interval 1 --fix_random_seed
```

Then evaluate:

```bash
cd runner
CUDA_VISIBLE_DEVICES=0,1 python test.py --batch_size 64 --ckpt /absolute/path/to/best_model.pth --val --full_eval --ema_coef 0.999 --extra_tag h200_repro_val
```

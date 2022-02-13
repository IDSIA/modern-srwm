# Few-Shot Learning & Sequential Adaptation Experiments

This directory contains code we used for few-shot learning and multi-task sequential adaptation experiments.

NB: Separate license files can be found for the following code.
* `torchmeta_local` directory contains a fork of [torchmeta](https://github.com/tristandeleu/pytorch-meta) which we modified locally.
* `resnet_impl.py` is originally forked from [yinboc/few-shot-meta-baseline](https://github.com/yinboc/few-shot-meta-baseline/blob/master/models/resnet12.py).

There are four main files:
* `main_few_shot_sync.py` & `eval_sync.py` to train/eval models for few-shot learning (synchronous label setting), 
* `main_few_shot_delayed_multi_sequential.py` & `eval_delay_multi_sequential.py` to train/eval models for multi-task sequential adaptation (delayed label setting)

Data files will be automatically downloaded to the folder specified via `--data_dir`.
See examples below for training and evaluation.
In all cases, a single GPU will be used.

## Requirements
* Required packages can be found in `requirements.txt`
* PyTorch >= 1.10 is recommended for the sequential adaptation experiments (`main_few_shot_delayed_multi_sequential.py`), especially for evaluation (`eval_delay_multi_sequential.py`), as we use `stable=True` option for `torch.sort` which only works on GPUs from 1.10 (according to its documentation).
* `ninja` is needed to compile custom CUDA kernels (`self_ref_v0/self_ref_v0.cu` & `fast_weight/fast_weight_cuda.cu`).
* Optionally: `wandb` for monitoring jobs (by using the `--use_wandb` flag)

## Standard Few-Shot Learning (Synchronous Label Setting):
### Training:
```
export TORCH_EXTENSIONS_DIR="/home/me/torch_extensions_fsl"

python3 ./main_few_shot_sync.py \
  --name_dataset omniglot \
  --seed 1 \
  --num_worker 8 \
  --model_type 'srwm' \
  --work_dir save_models \
  --total_train_steps 300_000 \
  --validate_every 10000 \
  --batch_size 128 \
  --num_layer 2 \
  --ff_factor 4 \
  --k_shot 1 \
  --hidden_size 256 \
  --n_head 16 \
  --test_per_class 1 \
  --learning_rate 1e-3 \
```
### Evaluation:
```
export TORCH_EXTENSIONS_DIR="/home/me/torch_extensions_fsl"

python3 ./eval_sync.py \
  --name_dataset omniglot \
  --load_from_checkpoint save_models/20420211-124941/best_model.pt \
  --num_test 5 \
  --test_size 1000 \
  --seed 1 \
  --num_worker 8 \
  --model_type 'srwm' \
  --work_dir save_models \
  --batch_size 16 \
  --num_layer 2 \
  --ff_factor 4 \
  --k_shot 1 \
  --hidden_size 256 \
  --n_head 16 \
  --test_per_class 1 \
```

## Sequential Multi-Task Adaptation (Delayed Label Setting):
### Training:
```
export TORCH_EXTENSIONS_DIR="/home/me/torch_extensions_fsl"

python3 ./main_few_shot_delayed_multi_sequential.py \
  --seed 1 \
  --num_worker 8 \
  --model_type 'srwm' \
  --work_dir save_models \
  --total_train_steps 1_000_000 \
  --validate_every 4000 \
  --batch_size 32 \
  --grad_cummulate 2 \
  --num_layer 3 \
  --hidden_size 256 \
  --n_head 16 \
  --k_shot 15 \
  --max_trim 60 \
  --ff_factor 1 \
  --learning_rate 3e-4 \
  --test_per_class 1 \
  --use_wandb \
  --project_name 'multi_sequential'
```

### Evaluation:
```
export TORCH_EXTENSIONS_DIR="/home/me/torch_extensions_fsl"

python3 ./eval_delay_multi_sequential.py \
  --seed 1 \
  --load_from save_models/20420211-173326/ \
  --num_worker 8 \
  --model_type 'srwm' \
  --work_dir save_models \
  --batch_size 16 \
  --num_layer 3 \
  --ff_factor 1 \
  --n_head 16 \
  --hidden_size 256 \
  --k_shot 15 \
  --test_per_class 1 \
```

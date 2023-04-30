#!/bin/bash

# `--num_future_shot` number of "future shots" for bootstrapping

SEED=1

export TORCH_EXTENSIONS_DIR=

CODE=
DATA=

python3 ${CODE}/main_few_shot_sync_bootstrapping.py \
  --data_dir ${DATA} \
  --name_dataset miniimagenet \
  --seed ${SEED} \
  --num_worker 8 \
  --num_future_shot 5 \
  --test_per_class 1 \
  --model_type 'stateful_srwm' \
  --use_kl_loss \
  --work_dir save_models \
  --total_epoch 2 \
  --total_train_steps 600_000 \
  --validate_every 1_000 \
  --batch_size 16 \
  --num_layer 3 \
  --n_head 16 \
  --hidden_size 256 \
  --ff_factor 8 \
  --dropout 0.1 \
  --vision_dropout 0.1 \
  --k_shot 10 \
  --test_per_class 1 \
  --main_loss_scaler 1 \
  --bstp_loss_scaler 10 \
  --future_loss_scaler 1 \
  --use_warmup \
  --project_name 'btsp_miniimagenet_10shot' \

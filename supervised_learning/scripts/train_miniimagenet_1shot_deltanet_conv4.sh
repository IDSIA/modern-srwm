#!/bin/bash

SEED=1

export TORCH_EXTENSIONS_DIR=

CODE=
DATA=

python3 ${CODE}/main_few_shot_sync.py \
  --data_dir ${DATA} \
  --name_dataset miniimagenet \
  --seed ${SEED} \
  --num_worker 12 \
  --model_type 'deltanet' \
  --work_dir save_models \
  --total_epoch 2 \
  --total_train_steps 600_000 \
  --validate_every 1_000 \
  --batch_size 16 \
  --num_layer 4 \
  --n_head 16 \
  --hidden_size 256 \
  --ff_factor 8 \
  --dropout 0.1 \
  --vision_dropout 0.3 \
  --k_shot 1 \
  --test_per_class 1 \
  --learning_rate 1e-4 \
  --project_name 'miniimagenet_1shot' \
  --use_wandb

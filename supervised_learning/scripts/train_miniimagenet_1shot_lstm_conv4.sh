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
  --model_type 'lstm' \
  --work_dir save_models \
  --total_epoch 4 \
  --total_train_steps 1_200_000 \
  --validate_every 1_000 \
  --batch_size 16 \
  --num_layer 1 \
  --hidden_size 1024 \
  --dropout 0.0 \
  --vision_dropout 0.1 \
  --k_shot 1 \
  --test_per_class 1 \
  --learning_rate 3e-4 \
  --project_name 'miniimagenet_5shot' \
  --use_wandb

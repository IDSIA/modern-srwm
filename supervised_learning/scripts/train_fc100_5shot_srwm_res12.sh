#!/bin/bash

SEED=1

export TORCH_EXTENSIONS_DIR=

CODE=
DATA=

python3 ${CODE}/main_few_shot_sync.py \
  --data_dir ${DATA} \
  --name_dataset fc100 \
  --seed ${SEED} \
  --num_worker 12 \
  --model_type 'res12_srwm' \
  --work_dir save_models \
  --total_epoch 2 \
  --total_train_steps 600_000 \
  --validate_every 1_000 \
  --batch_size 16 \
  --num_layer 3 \
  --n_head 16 \
  --hidden_size 256 \
  --ff_factor 4 \
  --dropout 0.3 \
  --vision_dropout 0.3 \
  --dropout_type '2d_inblock' \
  --k_shot 5 \
  --test_per_class 1 \
  --use_warmup \
  --project_name 'fc100_5shot' \
  --use_wandb

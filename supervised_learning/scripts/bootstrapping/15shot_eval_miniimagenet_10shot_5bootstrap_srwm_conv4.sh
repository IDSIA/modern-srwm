#!/bin/bash

# change `--k_shot` to change K_test

SEED=1

export TORCH_EXTENSIONS_DIR=

CODE=
DATA=

python3 ${CODE}/eval_sync.py \
  --data_dir ${DATA} \
  --load_from_checkpoint pretrained_miniimagenet_10shot_5bootstrap_srwm_conv4.pt \
  --name_dataset miniimagenet \
  --seed ${SEED} \
  --num_worker 8 \
  --test_per_class 1 \
  --model_type 'stateful_srwm' \
  --work_dir save_models \
  --batch_size 32 \
  --num_layer 3 \
  --n_head 16 \
  --hidden_size 256 \
  --ff_factor 8 \
  --dropout 0.1 \
  --vision_dropout 0.1 \
  --k_shot 15 \
  --test_per_class 1 \
  --num_test 10 \
  --test_size 500

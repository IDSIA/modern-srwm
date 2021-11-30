# ProcGen Multi-Task RL Experiments

This repository was originally forked from [IDSIA/recurrent-fwp/reinforcement_learning](https://github.com/IDSIA/recurrent-fwp/tree/master/reinforcement_learning)
which itself is a fork of [Torchbeast](https://github.com/facebookresearch/torchbeast) (we only support the `polybeast` version).

Please refer to the original repositories for general instructions about installation, requirements, etc.

The main changes we introduced for this work are:
* Support of [ProcGen environments](https://openai.com/blog/procgen-benchmark/) (single & multi-task settings)
* Implementation of agents using self-referential weight matrices (including two CUDA kernels: [torchbeast/self_ref_v0/self_ref_v0.cu](https://github.com/IDSIA/modern-srwm/blob/main/reinforcement_learning/torchbeast/self_ref_v0/self_ref_v0.cu) and [torchbeast/self_ref_v1/self_ref_v1.cu](https://github.com/IDSIA/modern-srwm/blob/main/reinforcement_learning/torchbeast/self_ref_v1/self_ref_v1.cu))

See examples below for training and evaluation.

## Training 

Here is an example training command:

```
export TORCH_EXTENSIONS_DIR="/home/me/torchbeast_kernels"
export CUDA_VISIBLE_DEVICES=3

SAVE_DIR=saved_models

python -m torchbeast_procgen.polybeast \
     --single_gpu \
     --multi_env 6 \
     --pipes_basename "unix:/tmp/polybeast_somename" \
     --num_actors 48 \
     --num_servers 48 \
     --total_steps 300_000_000 \
     --save_extra_checkpoint 150_000_000 \
     --learning_rate 0.0006 \
     --grad_norm_clipping 40 \
     --epsilon 0.01 \
     --entropy_cost 0.01 \
     --batch_size 32 \
     --unroll_length 50 \
     --num_actions 15 \
     --num_layers 2 \
     --hidden_size 128 \
     --num_head 8 \
     --dim_head 16 \
     --use_sr \
     --num_learner_threads 1 \
     --num_inference_threads 1 \
     --project_name "proj_name_for_wandb" \
     --use_wandb \
     --xpid "experiment_name" \
     --num_levels 200 \
     --start_level 0 \
     --distribution_mode "easy" \
     --valid_distribution_mode "easy" \
     --valid_num_levels 200 \
     --valid_start_level 500 \
     --valid_num_episodes 10 \
     --savedir ${SAVE_DIR}
```

`--multi_env` specifies the number of environments N to be used for multi-task training.
It takes N items from the top of the list `list_procgen_env` in [torchbeast_procgen/polybeast_learner.py](https://github.com/IDSIA/modern-srwm/blob/main/reinforcement_learning/torchbeast_procgen/polybeast_learner.py).
To train in a single environment, remove this flag and instead add `--env procgen:procgen-${GAME}-v0`
where `GAME` should be one of the ProcGen games (see [list_procgen_games.txt](https://github.com/IDSIA/modern-srwm/blob/main/reinforcement_learning/list_procgen_games.txt)).

In our experiments, we did not make use of any validation set information, so any settings related to validation can be ignored in principle (by setting `--disable_validation`).

For multi-task training in the memory distribution, `distribution_mode` and `valid_distribution_mode` should be modified to `memory`
(and in our experiments, we set `num_levels` to 500 and `multi_env` to 4).

To specify the model to be used, provide one of the following flags:

* No flag for the feedforward baseline
* `--use_lstm` for the LSTM
* `--use_delta` for the Delta Net
* `--use_psr` for the 'Fake SR' model in the paper
* `--use_sr` for the SRWM model in the paper (example above)
* `--use_smfwp` for the SR-Delta model in the paper (used in the memory env. experiments).
* `--use_no_carry_sr` for SRWM with state reset (used in the ablation study)

(there are many other models implemented which were not used in this work).

Depending on the machine, we experienced that prepending `CUDA_LAUNCH_BLOCKING=1` was necessary to run the training script above.

## Evaluation

For evaluation, directly call `torchbeast_procgen.polybeast_learner` with `--mode test` flag.

Here is an example evaluation command:
```
export TORCH_EXTENSIONS_DIR="/home/me/torchbeast_kernels"
export CUDA_VISIBLE_DEVICES=3

GAME="starpilot"

SAVE_DIR=saved_models
MODEL_DIR=experiment_name

echo "Evaluation for ${GAME} ========"

python -m torchbeast_procgen.polybeast_learner \
     --mode test \
     --env procgen:procgen-${GAME}-v0 \
     --num_actors 48 \
     --num_servers 48 \
     --total_steps 200000000 \
     --save_extra_checkpoint 50000000 \
     --learning_rate 0.0006 \
     --grad_norm_clipping 40 \
     --epsilon 0.01 \
     --entropy_cost 0.01 \
     --batch_size 32 \
     --unroll_length 50 \
     --num_actions 15 \
     --num_layers 2 \
     --hidden_size 128 \
     --num_head 8 \
     --dim_head 16 \
     --use_sr \
     --num_learner_threads 1 \
     --num_inference_threads 1 \
     --xpid ${MODEL_DIR} \
     --num_levels 200 \
     --start_level 0 \
     --test_distribution_mode 'easy' \
     --test_num_levels 200 \
     --test_start_level 0 \
     --savedir ${SAVE_DIR}
```
where `GAME` should be modified to specify the test game (see [list_procgen_games.txt](https://github.com/IDSIA/modern-srwm/blob/main/reinforcement_learning/list_procgen_games.txt)), `test_num_levels` and `test_start_level` should be modified to specify the split for evaluation.

For testing performance on the training set :
```
     --test_num_levels 200 \
     --test_start_level 0 \
```

We used the following 3 test splits:

Test 1:
```
     --test_num_levels 200 \
     --test_start_level 1000 \
```

Test 2:
```
     --test_num_levels 200 \
     --test_start_level 1200 \
```

Test 3:
```
     --test_num_levels 200 \
     --test_start_level 1400 \
```

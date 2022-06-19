# main file to be executed to train models in sequential multi-task few shot
# learning

import os
import sys
import json
import time
from datetime import datetime
import argparse
import logging
import numpy as np
import random

import torch
import torch.nn as nn
from torchmeta_local.utils.data import BatchMetaDataLoader
from torchmeta_local.datasets.helpers import omniglot_rgb84x84_norm
from torchmeta_local.datasets.helpers import miniimagenet_norm

from model_few_shot import (
    ConvLSTMModel, ConvDeltaModel, ConvSRWMModel,
    Res12LSTMModel, Res12DeltaModel, Res12SRWMModel)
from utils_few_shot import eval_model_delayed_label_multi_sequential


parser = argparse.ArgumentParser(
    description='Sequential multi-task adaptation.')
parser.add_argument('--data_dir', type=str,
                    default='./data', help='location of the data corpus')
parser.add_argument('--name_dataset', type=str, default='miniimagenet_norm',
                    choices=['miniimagenet_norm'])
parser.add_argument('--num_worker', default=12, type=int,
                    help='for dataloader.')
parser.add_argument('--work_dir', default='save_models', type=str,
                    help='where to save model ckpt.')
parser.add_argument('--init_model_from', default=None, type=str,
                    help='e.g. save_models/aaa/best_model.pt.')
parser.add_argument('--model_type', type=str, default='lstm',
                    choices=['lstm', 'deltanet', 'srwm',
                             'res12_lstm', 'res12_deltanet', 'res12_srwm'],
                    help='model architecture')
parser.add_argument('--seed', default=1, type=int, help='Seed.')
parser.add_argument('--valid_seed', default=0, type=int, help='Seed.')
parser.add_argument('--test_seed', default=0, type=int, help='Seed.')
parser.add_argument('--disable_eval_shuffling', action='store_true',
                    help='disable shuffling of valid/test sets. Only useful '
                         'to reproduce old/buggy behavior.')
# model hyper-parameters:
parser.add_argument('--num_layer', default=1, type=int,
                    help='number of layers. for both LSTM and Trafo.')
parser.add_argument('--hidden_size', default=512, type=int,
                    help='hidden size. for both LSTM and Trafo.')
parser.add_argument('--n_head', default=8, type=int,
                    help='Transformer number of heads.')
parser.add_argument('--ff_factor', default=4, type=int,
                    help='Transformer ff dim to hidden dim ratio.')
parser.add_argument('--dropout', default=0.0, type=float,
                    help='dropout rate.')
parser.add_argument('--vision_dropout', default=0.0, type=float,
                    help='dropout rate in the vision feat extractor.')
parser.add_argument('--srwm_beta_init', default=0.0, type=float,
                    help='beta bias for srwm.')
parser.add_argument('--use_input_softmax', action='store_true',
                    help='input softmax for srwm.')

# few shot learning setting
parser.add_argument('--n_way', default=5, type=int,
                    help='number of possible classes per train/test episode.')
parser.add_argument('--k_shot', default=15, type=int,
                    help='number of examples in the `train` part of torchmeta')
parser.add_argument('--test_per_class', default=1, type=int,
                    help='param for torchmeta; number of query examples')
parser.add_argument('--max_trim', default=None, type=int,
                    help='maximum number of positions to be removed. if None, '
                         'computed based on `n_way` and `k_shot`.')

# training hyper-parameters:
parser.add_argument('--total_train_steps', default=100000, type=int,
                    help='Number of training steps to train on')
parser.add_argument('--valid_size', default=100, type=int,
                    help='Number of valid batches to validate on')
parser.add_argument('--test_size', default=100, type=int,
                    help='Number of test batches to test on')
parser.add_argument('--batch_size', default=16, type=int,
                    help='batch size.')
parser.add_argument('--learning_rate', default=1e-3, type=float,
                    help='batch size.')
parser.add_argument('--grad_cummulate', default=1, type=int,
                    help='number of gradient accumulation steps.')
parser.add_argument('--report_every', default=100, type=int,
                    help='Report log every this steps (not used).')
parser.add_argument('--validate_every', default=1000, type=int,
                    help='Report log every this steps (not used).')
parser.add_argument('--clip', default=0.0, type=float,
                    help='global norm clipping threshold.')
# for wandb
parser.add_argument('--project_name', type=str, default=None,
                    help='project name for wandb.')
parser.add_argument('--job_name', type=str, default=None,
                    help='job name for wandb.')
parser.add_argument('--use_wandb', action='store_true',
                    help='use wandb.')

args = parser.parse_args()

model_name = args.model_type

# Set work directory
args.work_dir = os.path.join(args.work_dir, time.strftime('%Y%m%d-%H%M%S'))
if not os.path.exists(args.work_dir):
    os.makedirs(args.work_dir)

work_dir_key = '/'.join(os.path.abspath(args.work_dir).split('/')[-3:])

# logging
log_file_name = f"{args.work_dir}/log.txt"
handlers = [logging.FileHandler(log_file_name), logging.StreamHandler()]
logging.basicConfig(
    level=logging.INFO, format='%(message)s', handlers=handlers)

loginf = logging.info

loginf(f"torch version: {torch.__version__}")
loginf(f"Work dir: {args.work_dir}")

# wandb settings
if args.use_wandb:  # configure wandb.
    import wandb
    use_wandb = True

    if args.project_name is None:
        project_name = (os.uname()[1]
                        + datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    else:
        project_name = args.project_name

    wandb.init(
        project=project_name, settings=wandb.Settings(start_method='fork'))
    # or `settings=wandb.Settings(start_method='thread')`
    if args.job_name is None:
        wandb.run.name = f"{os.uname()[1]}//" \
                         f"{model_name}-{args.name_dataset}//" \
                         f"seed{args.seed}//" \
                         f"{args.test_per_class}-test_per_cl/" \
                         f"{args.n_way}way-{args.k_shot}shot-" \
                         f"{args.max_trim}trim/" \
                         f"L{args.num_layer}/h{args.hidden_size}/" \
                         f"n{args.n_head}/ff{args.ff_factor}/" \
                         f"d{args.dropout}/vd{args.vision_dropout}/" \
                         f"b{args.batch_size}/" \
                         f"lr{args.learning_rate}/" \
                         f"g{args.grad_cummulate}/bias{args.srwm_beta_init}" \
                         f"softmax{args.use_input_softmax}" \
                         f"//PATH'{work_dir_key}'//"
    else:
        wandb.run.name = f"{os.uname()[1]}//{args.job_name}"

    config = wandb.config
    config.host = os.uname()[1]  # host node name
    config.seed = args.seed
    config.test_per_class = args.test_per_class
    config.n_way = args.n_way
    config.k_shot = args.k_shot
    config.max_trim = args.max_trim
    config.srwm_beta_init = args.srwm_beta_init
    config.use_input_softmax = args.use_input_softmax
    config.name_dataset = args.name_dataset
    config.work_dir = args.work_dir
    config.model_type = args.model_type
    config.hidden_size = args.hidden_size
    config.n_head = args.n_head
    config.ff_factor = args.ff_factor
    config.dropout = args.dropout
    config.vision_dropout = args.vision_dropout
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.grad_cummulate = args.grad_cummulate
    config.report_every = args.report_every
    config.disable_eval_shuffling = args.disable_eval_shuffling
else:
    use_wandb = False
# end wandb

# save args
loginf(f"Command executed: {sys.argv[:]}")
loginf(f"Args: {json.dumps(args.__dict__, indent=2)}")

with open(f'{args.work_dir}/args.txt', 'w') as f:
    json.dump(args.__dict__, f, indent=2)

# set seed
loginf(f"Seed: {args.seed}")
seed = args.seed
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

valid_seed = args.valid_seed
test_seed = args.test_seed
loginf(f"Valid seed: {valid_seed}, Test seed: {test_seed}")
shuffled_eval = not args.disable_eval_shuffling

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# set dataset
batch_size = args.batch_size
n_way = args.n_way
k_shot_train = args.k_shot
test_per_class = args.test_per_class

loginf(f"Dataset/Task: omniglot + miniimagenet_norm")

task_id_to_name = {0: 'omniglot', 1: 'imagenet'}

# Omniglot
omniglot_dataset = omniglot_rgb84x84_norm(
    args.data_dir, ways=n_way, shots=k_shot_train, test_shots=test_per_class,
    meta_train=True, download=True, shuffle=True, seed=seed)
omniglot_dataloader = BatchMetaDataLoader(
    omniglot_dataset, batch_size=batch_size // 2, num_workers=args.num_worker,
    pin_memory=True)

omniglot_val_dataset = omniglot_rgb84x84_norm(
    args.data_dir, ways=n_way, shots=k_shot_train, test_shots=test_per_class,
    meta_val=True, shuffle=shuffled_eval, seed=valid_seed)
omniglot_val_dataloader = BatchMetaDataLoader(
    omniglot_val_dataset, batch_size=batch_size // 2,
    num_workers=args.num_worker, pin_memory=True)

# Mini-imagenet
imagenet_dataset = miniimagenet_norm(
    args.data_dir, ways=n_way, shots=k_shot_train, test_shots=test_per_class,
    meta_train=True, download=True, shuffle=True, seed=seed)
imagenet_dataloader = BatchMetaDataLoader(
    imagenet_dataset, batch_size=batch_size // 2, num_workers=args.num_worker,
    pin_memory=True)

imagenet_val_dataset = miniimagenet_norm(
    args.data_dir, ways=n_way, shots=k_shot_train, test_shots=test_per_class,
    meta_val=True, shuffle=shuffled_eval, seed=valid_seed)
imagenet_val_dataloader = BatchMetaDataLoader(
    imagenet_val_dataset, batch_size=batch_size // 2,
    num_workers=args.num_worker, pin_memory=True)

val_dataloader = {
    'omniglot': omniglot_val_dataloader,
    'miniimagenet': imagenet_val_dataloader,
}

device = 'cuda'

# setting model

if args.max_trim is None:
    assert args.k_shot > 6, f'k_shot too small {args.k_shot}'
    max_trim = args.k_shot - 6  # to see at least 5 shot performance
else:
    max_trim = args.max_trim

hidden_size = args.hidden_size
num_classes = args.n_way

num_layer = args.num_layer
n_head = args.n_head
dim_head = hidden_size // n_head
dim_ff = hidden_size * args.ff_factor
dropout_rate = args.dropout
vision_dropout = args.vision_dropout

is_imagenet = args.name_dataset != 'omniglot'

if model_name == 'lstm':  # conv lstm
    loginf("Model: LSTM")
    model = ConvLSTMModel(hidden_size, num_classes, num_layer=num_layer,
                          vision_dropout=vision_dropout,
                          imagenet=is_imagenet)
elif model_name == 'deltanet':
    loginf("Model: DeltaNet")
    model = ConvDeltaModel(hidden_size=hidden_size, num_layers=num_layer,
                           num_head=n_head, dim_head=dim_head, dim_ff=dim_ff,
                           dropout=dropout_rate, num_classes=num_classes,
                           vision_dropout=vision_dropout,
                           imagenet=is_imagenet)
elif model_name == 'srwm':
    loginf("Model: Self-Referential learning")
    model = ConvSRWMModel(hidden_size=hidden_size, num_layers=num_layer,
                          num_head=n_head, dim_head=dim_head, dim_ff=dim_ff,
                          dropout=dropout_rate, num_classes=num_classes,
                          vision_dropout=vision_dropout,
                          use_ln=True, beta_init=args.srwm_beta_init,
                          use_input_softmax=args.use_input_softmax,
                          imagenet=is_imagenet)

elif model_name == 'res12_lstm':
    loginf("Model: Resnet12 + LSTM")
    model = Res12LSTMModel(hidden_size=hidden_size, num_layers=num_layer,
                           dropout=dropout_rate,
                           vision_dropout=vision_dropout,
                           num_classes=num_classes, imagenet=is_imagenet)
elif model_name == 'res12_deltanet':
    assert is_imagenet, 'Mainly for Imagenet'
    loginf("Model: Resnet12 + Deltanet")
    model = Res12DeltaModel(hidden_size=hidden_size, num_layers=num_layer,
                            num_head=n_head, dim_head=dim_head, dim_ff=dim_ff,
                            dropout=dropout_rate, num_classes=num_classes,
                            vision_dropout=vision_dropout,
                            imagenet=is_imagenet)
elif model_name == 'res12_srwm':
    assert is_imagenet, 'Mainly for Imagenet'
    loginf("Model: Resnet12 + SRWM")
    model = Res12SRWMModel(hidden_size=hidden_size, num_layers=num_layer,
                           num_head=n_head, dim_head=dim_head, dim_ff=dim_ff,
                           dropout=dropout_rate, num_classes=num_classes,
                           vision_dropout=vision_dropout,
                           use_ln=True, beta_init=args.srwm_beta_init,
                           use_input_softmax=args.use_input_softmax,
                           imagenet=is_imagenet)

loginf(f"Number of trainable params: {model.num_params()}")
loginf(f"{model}")

model = model.to(device)

# load if needed
if args.init_model_from is not None:
    loginf(f"loading model from {args.init_model_from}")
    checkpoint = torch.load(args.init_model_from)
    model.load_state_dict(checkpoint['model_state_dict'])

# Set optimiser
learning_rate = args.learning_rate
clip = args.clip

loginf(f"Learning rate: {learning_rate}")
loginf(f"clip at: {clip}")

loginf(f"Batch size: {args.batch_size}")
loginf(f"Gradient accumulation for {args.grad_cummulate} steps.")

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             betas=(0.9, 0.995), eps=1e-9)

model.reset_grad()
############

best_model_path = os.path.join(args.work_dir, 'best_model.pt')
lastest_model_path = os.path.join(args.work_dir, 'lastest_model.pt')

loginf(f"[{datetime.now().strftime('%Y/%m/%d %H:%M:%S')}] Start training")
start_time = time.time()
interval_start_time = time.time()
train_timer = time.time()
last_batch_logged = 0

acc_per_shot = {0: [], 1: []}
cnt_per_shot = {0: [], 1: []}

for key in acc_per_shot.keys():
    for shot in range(k_shot_train):
        acc_per_shot[key].append(0)
        cnt_per_shot[key].append(0)

best_total_val_acc = 0.0

num_seq = 0
running_loss = 0.0
running_total = 0
running_correct = 0
run_step = 0

task_running_correct = {
    'omniglot': 0.,
    'miniimagenet': 0.
}

counts = {
    'omniglot': 0.,
    'miniimagenet': 0.
}


for i, (omni_batch, imagenet_batch) in enumerate(zip(omniglot_dataloader, imagenet_dataloader)):
    model.train()
    state = None

    # Omniglot
    om_train_inputs, om_train_targets = omni_batch['train']
    im_train_inputs, im_train_targets = imagenet_batch['train']
    del omni_batch['test'], imagenet_batch['test']

    om_train_inputs = om_train_inputs.to(device=device)
    om_train_targets = om_train_targets.to(device=device)  # (B, len)

    om_train_inputs = om_train_inputs.transpose(0, 1)  # (len, B, **)
    om_train_targets = om_train_targets.transpose(0, 1)  # (len, B)

    # randomly remove n last positions
    trim_offset = random.randint(0, max_trim)
    if trim_offset > 0:
        om_train_inputs = om_train_inputs[:-trim_offset]
        om_train_targets = om_train_targets[:-trim_offset]

    om_len, om_bsz = om_train_targets.shape
    num_seq += om_bsz

    # Imagenet
    im_train_inputs = im_train_inputs.to(device=device)  # (B, len, **)
    im_train_targets = im_train_targets.to(device=device)  # (B, len)

    im_train_inputs = im_train_inputs.transpose(0, 1)  # (len, B, **)
    im_train_targets = im_train_targets.transpose(0, 1)  # (len, B)

    # randomly remove n last positions
    trim_offset = random.randint(0, max_trim)
    if trim_offset > 0:
        im_train_inputs = im_train_inputs[:-trim_offset]
        im_train_targets = im_train_targets[:-trim_offset]

    im_len, im_bsz = im_train_targets.shape
    num_seq += im_bsz

    # contenate along time dimension, randomize order for each batch
    order_ = random.randint(0, 1)  # 2 is inclusive!
    if order_ == 0:  # omniglot first
        net_input = torch.cat([om_train_inputs, im_train_inputs], dim=0)
        target_labels = torch.cat([om_train_targets, im_train_targets], dim=0)
    else:  # miniimagenet first
        net_input = torch.cat([im_train_inputs, om_train_inputs], dim=0)
        target_labels = torch.cat([im_train_targets, om_train_targets], dim=0)

    slen, bsz = target_labels.shape
    assert bsz == im_bsz == om_bsz

    delayed_labels = target_labels[:-1]
    dummy_last_token = torch.zeros_like(delayed_labels[0].unsqueeze(0))
    label_feedback = torch.cat([dummy_last_token, delayed_labels], dim=0)

    outputs, _ = model(net_input, label_feedback)
    outputs = outputs.reshape(slen * bsz, num_classes)

    target_labels = target_labels.reshape(-1)
    loss = loss_fn(outputs, target_labels)
    loss.backward()

    if i % args.grad_cummulate == 0:
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        model.reset_grad()

    # global loss
    running_loss += loss.item()
    running_total += slen * bsz
    model.eval()

    with torch.no_grad():
        _, predicted = outputs.max(-1)
    bool_correct_pred = (predicted == target_labels)  # (slen * B)
    running_correct += bool_correct_pred.sum().item()

    target_labels = target_labels.reshape(slen, bsz)
    bool_correct_pred = bool_correct_pred.reshape(slen, bsz)

    if order_ == 0:  # omniglot first
        om_bool_correct_pred = bool_correct_pred[:om_len]
        im_bool_correct_pred = bool_correct_pred[om_len:]
    else:  # imagenet first
        im_bool_correct_pred = bool_correct_pred[:im_len]
        om_bool_correct_pred = bool_correct_pred[im_len:]

    task_running_correct['omniglot'] += om_bool_correct_pred.sum().item()
    task_running_correct['miniimagenet'] += im_bool_correct_pred.sum().item()

    counts['omniglot'] += om_len * om_bsz
    counts['miniimagenet'] += im_len * im_bsz

    om_train_targets = om_train_targets.transpose(0, 1)  # B, len
    om_bool_correct_pred = om_bool_correct_pred.transpose(0, 1)  # B, len
    im_train_targets = im_train_targets.transpose(0, 1)  # B, len
    im_bool_correct_pred = im_bool_correct_pred.transpose(0, 1)  # B, len

    for b in range(bsz):
        # omniglot
        prev_cl_end = 0
        _, cnts_uniq = torch.unique(
            om_train_targets[b], sorted=True, return_counts=True)
        _, indices = torch.sort(om_train_targets[b], stable=True)
        cnts_uniq_len = len(cnts_uniq)
        for cl in range(n_way):
            if cl < cnts_uniq_len:
                cl_cnts = cnts_uniq[cl]
                cl_indices = indices[prev_cl_end:prev_cl_end + cl_cnts]
                cl_indices_len = len(cl_indices)
                prev_cl_end += cl_cnts

                for shot in range(k_shot_train):
                    if cl_indices_len > shot:
                        acc_per_shot[0][shot] += (
                            om_bool_correct_pred[b][cl_indices[shot]].item())
                        cnt_per_shot[0][shot] += 1
        # imagenet
        prev_cl_end = 0
        _, cnts_uniq = torch.unique(
            im_train_targets[b], sorted=True, return_counts=True)
        _, indices = torch.sort(im_train_targets[b], stable=True)
        cnts_uniq_len = len(cnts_uniq)
        for cl in range(n_way):
            if cl < cnts_uniq_len:
                cl_cnts = cnts_uniq[cl]
                cl_indices = indices[prev_cl_end:prev_cl_end + cl_cnts]
                cl_indices_len = len(cl_indices)
                prev_cl_end += cl_cnts

                for shot in range(k_shot_train):
                    if cl_indices_len > shot:
                        acc_per_shot[1][shot] += (
                            im_bool_correct_pred[b][cl_indices[shot]].item())
                        cnt_per_shot[1][shot] += 1

    run_step += 1
    if i % args.report_every == 0:
        om_train_ac = task_running_correct['omniglot'] / counts['omniglot']
        im_train_ac = (
            task_running_correct['miniimagenet'] / counts['miniimagenet'])
        if use_wandb:
            wandb_log = {}
            wandb_log["train_loss"] = running_loss / run_step
            wandb_log["running_acc"] = 100 * running_correct / running_total
            wandb_log["omniglot_train_acc"] = 100 * om_train_ac
            wandb_log["imagenet_train_acc"] = 100 * im_train_ac
            for key in acc_per_shot.keys():
                for shot in range(k_shot_train):
                    if cnt_per_shot[key][shot] > 0:
                        shot_acc = (
                            100 * acc_per_shot[key][shot] / cnt_per_shot[key][shot]
                        )
                    else:
                        shot_acc = 0.0
                    wandb_log[f"{task_id_to_name[key]}_tr_{shot}"] = shot_acc
            wandb.log(wandb_log)

        train_elapsed = time.time() - train_timer
        train_timer = time.time()
        num_images_per_sec = (
            (i + 1 - last_batch_logged) * batch_size * slen // train_elapsed)
        last_batch_logged = i

        log_str = f'steps: {i}, num_seq: {num_seq}, '
        log_str += f'train_loss: {running_loss / run_step :.3f}, '
        log_str += (
            f'running_acc: {100 * running_correct / running_total:.2f} % ')
        log_str += (
            f'(elapsed {int(train_elapsed)}s, {int(num_images_per_sec)} '
            + 'images/s)')
        loginf(log_str)

        log_str = ''
        log_str += f'omniglot_train_acc: {100 * om_train_ac:.2f} % '
        log_str += f'imagenet_train_acc: {100 * im_train_ac:.2f} % '
        loginf(log_str)

        for key in acc_per_shot.keys():
            log_str = f'[{task_id_to_name[key]}] '
            for shot in range(k_shot_train):
                if cnt_per_shot[key][shot] > 0:
                    shot_acc = (
                        100 * acc_per_shot[key][shot] / cnt_per_shot[key][shot]
                    )
                else:
                    shot_acc = 0
                log_str += f"{key}_train_{shot}: {shot_acc:.2f} % "
            loginf(log_str)

        running_loss = 0.0
        running_total = 0
        running_correct = 0
        run_step = 0
        task_running_correct = {
            'omniglot': 0.,
            'miniimagenet': 0.,
        }
        counts = {
            'omniglot': 0.,
            'miniimagenet': 0.
        }
        acc_per_shot = {0: [], 1: []}
        cnt_per_shot = {0: [], 1: []}

        for key in acc_per_shot.keys():
            for _ in range(k_shot_train):
                acc_per_shot[key].append(0)
                cnt_per_shot[key].append(0)

    if i % args.validate_every == 0:  # run validation
        model.eval()
        # val_acc_dict = {}
        with torch.no_grad():
            v_total, task_wise_acc, val_acc_dict = (
                eval_model_delayed_label_multi_sequential(
                    model, val_dataloader['omniglot'],
                    val_dataloader['miniimagenet'],
                    n_way=n_way, k_shot=k_shot_train,
                    num_steps=args.valid_size))

        if use_wandb:
            wandb_log = {}
            wandb_log["val_acc"] = v_total
            wandb_log["omniglot_val_acc"] = task_wise_acc[0]
            wandb_log["imagenet_val_acc"] = task_wise_acc[1]
            for key in val_acc_dict.keys():
                for shot in range(k_shot_train):
                    wandb_log[f"{task_id_to_name[key]}_val_{shot}"] = (
                        val_acc_dict[key][shot])
            wandb.log(wandb_log)

        log_str = f"[val {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}] "
        loginf(log_str)
        for key in val_acc_dict.keys():
            log_str = ""
            log_str += f'{task_id_to_name[key]} val total {task_wise_acc[key]:.2f} %, '
            for shot in range(k_shot_train):
                log_str += f"val_{shot}: {val_acc_dict[key][shot]:.2f} %, "
            loginf(log_str)

        if v_total > best_total_val_acc:
            best_total_val_acc = v_total
            best_step = i
            # Save the best model
            loginf("The best model so far.")
            torch.save({'epoch': best_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'valid_acc': v_total}, best_model_path)
            loginf("Saved.")
        # Save the latest model
        torch.save({'train_step': i,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'valid_total_acc': v_total}, lastest_model_path)

        elapsed = time.time() - interval_start_time
        loginf(f"Elapsed {elapsed / 60.:.2f} min since last valid.")
        interval_start_time = time.time()
        train_timer = time.time()

    if i > args.total_train_steps:
        break

elapsed = time.time() - start_time
loginf(f"Finished {i} steps in {elapsed / 60.:.2f} min.")
loginf(f"Best one shot validation acc: {100 * best_total_val_acc:.2f} % "
       f"at step {best_step}")

# load the best model and evaluate on the test set
del (omniglot_dataset, omniglot_dataloader, omniglot_val_dataset,
     omniglot_val_dataloader, imagenet_dataset, imagenet_dataloader,
     imagenet_val_dataset, imagenet_val_dataloader)

omniglot_test_dataset = omniglot_rgb84x84_norm(
    args.data_dir, ways=n_way, shots=k_shot_train, test_shots=test_per_class,
    meta_test=True, download=True, shuffle=shuffled_eval, seed=test_seed)
omniglot_test_dataloader = BatchMetaDataLoader(
    omniglot_test_dataset, batch_size=batch_size // 2,
    num_workers=args.num_worker, pin_memory=True)

imagenet_test_dataset = miniimagenet_norm(
    args.data_dir, ways=n_way, shots=k_shot_train, test_shots=test_per_class,
    meta_test=True, download=True, shuffle=shuffled_eval, seed=test_seed)
imagenet_test_dataloader = BatchMetaDataLoader(
    imagenet_test_dataset, batch_size=batch_size // 2,
    num_workers=args.num_worker, pin_memory=True)

test_dataloader = {
    'omniglot': omniglot_test_dataloader,
    'miniimagenet': imagenet_test_dataloader,
}

checkpoint = torch.load(best_model_path)
model.load_state_dict(checkpoint['model_state_dict'])

test_acc_dict = {}
with torch.no_grad():
    v_total, task_wise_acc, val_acc_dict = (
        eval_model_delayed_label_multi_sequential(
            model, test_dataloader['omniglot'],
            test_dataloader['miniimagenet'],
            n_way=n_way, k_shot=k_shot_train, num_steps=args.test_size))

    log_str = f"[final test {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}] "
    loginf(log_str)
    for key in val_acc_dict.keys():
        log_str = ""
        log_str += f'{task_id_to_name[key]} val total {task_wise_acc[key]:.2f} %, '
        for shot in range(k_shot_train):
            log_str += f"val_{shot}: {val_acc_dict[key][shot]:.2f} %, "
        loginf(log_str)

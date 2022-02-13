# main file to be executed to evaluate models in sequential multi-task few shot
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
from torchmeta_local.utils.data import BatchMetaDataLoader
from torchmeta_local.datasets.helpers import omniglot_rgb84x84_norm
from torchmeta_local.datasets.helpers import miniimagenet_norm

from model_few_shot import (
    ConvLSTMModel, ConvDeltaModel, ConvSRWMModel,
    Res12LSTMModel, Res12DeltaModel, Res12SRWMModel)
from utils_few_shot import eval_per_pos_model_delayed_label_multi_sequential


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
parser.add_argument('--load_from', default='save_models/aaa/', type=str,
                    help='dir from where to load model ckpt.')
parser.add_argument('--model_type', type=str, default='lstm',
                    choices=['lstm', 'deltanet', 'srwm',
                             'res12_lstm', 'res12_deltanet', 'res12_srwm'],
                    help='model architecture')
parser.add_argument('--seed', default=1, type=int, help='Seed.')
parser.add_argument('--valid_seed', default=0, type=int, help='Seed.')
parser.add_argument('--test_seed', default=0, type=int, help='Seed.')

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
                    help='param for torchmeta')
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
parser.add_argument('--num_test', default=1, type=int,
                    help='Number of times we run test on random test set')
parser.add_argument('--imagenet_first', action='store_true',
                    help='imagenet then omniglot.')
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
    meta_val=True, shuffle=False, seed=valid_seed)  # fixed validation set
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
    meta_val=True, shuffle=False, seed=valid_seed)  # fixed validation set
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

best_model_path = os.path.join(args.load_from, 'best_model.pt')
lastest_model_path = os.path.join(args.load_from, 'lastest_model.pt')

loginf(f"[{datetime.now().strftime('%Y/%m/%d %H:%M:%S')}] Start Eval")
loginf(f"Loading model from {best_model_path}")
checkpoint = torch.load(best_model_path)
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()

omniglot_test_dataset = omniglot_rgb84x84_norm(
    args.data_dir, ways=n_way, shots=k_shot_train, test_shots=test_per_class,
    meta_test=True, download=True, shuffle=False, seed=test_seed)
omniglot_test_dataloader = BatchMetaDataLoader(
    omniglot_test_dataset, batch_size=batch_size // 2,
    num_workers=args.num_worker, pin_memory=True)

imagenet_test_dataset = miniimagenet_norm(
    args.data_dir, ways=n_way, shots=k_shot_train, test_shots=test_per_class,
    meta_test=True, download=True, shuffle=False, seed=test_seed)
imagenet_test_dataloader = BatchMetaDataLoader(
    imagenet_test_dataset, batch_size=batch_size // 2,
    num_workers=args.num_worker, pin_memory=True)

test_dataloader = {
    'omniglot': omniglot_test_dataloader,
    'miniimagenet': imagenet_test_dataloader,
}

num_test = args.num_test
test_size = args.test_size

omniglot_first = not args.imagenet_first

test_acc_dict = {}
with torch.no_grad():
    v_total, task_wise_acc, val_acc_dict, per_pos_acc = (
        eval_per_pos_model_delayed_label_multi_sequential(
            model, test_dataloader['omniglot'], test_dataloader['miniimagenet'],
            n_way=n_way, k_shot=k_shot_train, num_steps=args.test_size,
            omniglot_first=omniglot_first))

    log_str = f"[test {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}] "
    loginf(log_str)
    for key in val_acc_dict.keys():
        log_str = ""
        log_str += f'{task_id_to_name[key]} val total {task_wise_acc[key]:.2f} %, '
        for shot in range(k_shot_train):
            log_str += f"test_{shot}: {val_acc_dict[key][shot]:.2f} %, "
        loginf(log_str)

    log_str = 'Position-wise accuracy:'
    loginf(log_str)
    log_str = ""
    for pos in range(k_shot_train * n_way * 2):
        log_str += f"Acc_pos{pos}: {per_pos_acc[pos]:.2f} %, "
    loginf(log_str)

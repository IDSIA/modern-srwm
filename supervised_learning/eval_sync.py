# main file to be executed to evaluate models for few shot learning in the
# synchrous-label setting

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

from model_few_shot import (
    ConvLSTMModel, ConvDeltaModel, ConvSRWMModel,
    Res12LSTMModel, Res12DeltaModel, Res12SRWMModel)
from utils_few_shot import eval_model_label_sync


parser = argparse.ArgumentParser(
    description='N-way K-shot learning based on label synchronous '
                'seq-processing NNs with only predicting (N*K+1)th image.')
parser.add_argument('--data_dir', type=str,
                    default='./data', help='location of the data corpus')
parser.add_argument('--name_dataset', type=str, default='omniglot',
                    choices=['omniglot', 'miniimagenet', 'omniglot_rgb84x84',
                             'omniglot_rgb84x84_norm', 'omniglot_norm',
                             'miniimagenet_norm', 'fc100', 'fc100_norm'])
parser.add_argument('--num_worker', default=12, type=int,
                    help='for dataloader.')
parser.add_argument('--work_dir', default='save_models', type=str,
                    help='where to save model ckpt.')
parser.add_argument('--load_from', default='save_models/aaa/', type=str,
                    help='dir from where to load model ckpt.')
parser.add_argument('--load_from_checkpoint',
                    default=None,
                    type=str, help='path from where to load model ckpt.')
parser.add_argument('--model_type', type=str, default='lstm',
                    choices=['lstm', 'deltanet', 'srwm',
                             'res12_lstm', 'res12_deltanet', 'res12_srwm'],
                    help='model architecture')
parser.add_argument('--seed', default=1, type=int, help='Seed.')
parser.add_argument('--valid_seed', default=0, type=int, help='Seed.')
parser.add_argument('--test_seed', default=0, type=int, help='Seed.')
parser.add_argument('--fixed_test', action='store_true',
                    help='use fixed test set.')
parser.add_argument('--eval_on_valid', action='store_true',
                    help='use fixed test set.')

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
parser.add_argument('--k_shot', default=1, type=int,
                    help='number of examples in the `train` part of torchmeta')
parser.add_argument('--test_per_class', default=1, type=int,
                    help='param for torchmeta')

# training hyper-parameters:
parser.add_argument('--total_train_steps', default=100000, type=int,
                    help='Number of training steps to train on')
parser.add_argument('--valid_size', default=100, type=int,
                    help='Number of valid batches to validate on')
parser.add_argument('--test_size', default=100, type=int,
                    help='Number of test batches to test on')
parser.add_argument('--num_test', default=1, type=int,
                    help='Number of times we run test on random test set')
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

loginf(f"Dataset/Task: {args.name_dataset}")
if args.name_dataset == 'omniglot':
    from torchmeta_local.datasets.helpers import omniglot as data_cls
elif args.name_dataset == 'omniglot_norm':
    from torchmeta_local.datasets.helpers import omniglot_norm as data_cls
elif args.name_dataset == 'miniimagenet':
    from torchmeta_local.datasets.helpers import miniimagenet as data_cls
elif args.name_dataset == 'miniimagenet_norm':  # mean/std normalized
    from torchmeta_local.datasets.helpers import (
        miniimagenet_norm as data_cls)
elif args.name_dataset == 'omniglot_rgb84x84':
    from torchmeta_local.datasets.helpers import omniglot_rgb84x84 as data_cls
elif args.name_dataset == 'omniglot_rgb84x84_norm':  # mean/std normalized
    from torchmeta_local.datasets.helpers import (
        omniglot_rgb84x84_norm as data_cls)
elif args.name_dataset == 'fc100':
    from torchmeta_local.datasets.helpers import fc100 as data_cls
elif args.name_dataset == 'fc100_norm':
    from torchmeta_local.datasets.helpers import fc100_norm as data_cls
else:
    assert False, f'Unknown dataset: {args.name_dataset}'

# load test set
if args.eval_on_valid:
    test_dataset = data_cls(args.data_dir, ways=n_way, shots=k_shot_train,
                            test_shots=test_per_class, meta_val=True,
                            download=True, shuffle=False, seed=test_seed)
else:
    test_dataset = data_cls(args.data_dir, ways=n_way, shots=k_shot_train,
                            test_shots=test_per_class, meta_test=True,
                            download=True, shuffle=False, seed=test_seed)

if args.fixed_test:
    # https://github.com/tristandeleu/pytorch-meta/issues/132
    test_class_size = len(test_dataset.dataset)  # num classes in valid
    # `dataset` here is torchmeta ClassDataset
    import itertools
    from torch.utils.data import Subset
    cls_indices = np.array(range(test_class_size))
    all_indices = []
    for subset in itertools.combinations(cls_indices, args.n_way):
        all_indices.append(subset)
    test_total_size = args.test_size * batch_size
    test_indices = random.sample(all_indices, test_total_size)
    test_dataset = Subset(test_dataset, test_indices)

test_dataloader = BatchMetaDataLoader(
    test_dataset, batch_size=batch_size, num_workers=args.num_worker,
    pin_memory=True)

device = 'cuda'

# setting model

hidden_size = args.hidden_size
num_classes = args.n_way

num_layer = args.num_layer
n_head = args.n_head
dim_head = hidden_size // n_head
dim_ff = hidden_size * args.ff_factor
dropout_rate = args.dropout
vision_dropout = args.vision_dropout

# is_imagenet = args.name_dataset != 'omniglot'
is_imagenet = args.name_dataset not in ['omniglot', 'omniglot_norm']
is_fc100 = False

if args.name_dataset in ['fc100', 'fc100_norm']:
    is_fc100 = True
    is_imagenet = False

if model_name == 'lstm':  # conv lstm
    loginf("Model: LSTM")
    model = ConvLSTMModel(hidden_size, num_classes, num_layer=num_layer,
                          vision_dropout=vision_dropout,
                          imagenet=is_imagenet, fc100=is_fc100)
elif model_name == 'deltanet':
    loginf("Model: DeltaNet")
    model = ConvDeltaModel(hidden_size=hidden_size, num_layers=num_layer,
                           num_head=n_head, dim_head=dim_head, dim_ff=dim_ff,
                           dropout=dropout_rate, num_classes=num_classes,
                           vision_dropout=vision_dropout,
                           imagenet=is_imagenet, fc100=is_fc100)
elif model_name == 'srwm':
    loginf("Model: Self-Referential learning")
    model = ConvSRWMModel(hidden_size=hidden_size, num_layers=num_layer,
                          num_head=n_head, dim_head=dim_head, dim_ff=dim_ff,
                          dropout=dropout_rate, num_classes=num_classes,
                          vision_dropout=vision_dropout,
                          use_ln=True, beta_init=args.srwm_beta_init,
                          use_input_softmax=args.use_input_softmax,
                          imagenet=is_imagenet, fc100=is_fc100)
elif model_name == 'res12_lstm':
    loginf("Model: Resnet12 + LSTM")
    model = Res12LSTMModel(hidden_size=hidden_size, num_layers=num_layer,
                           dropout=dropout_rate,
                           vision_dropout=vision_dropout,
                           num_classes=num_classes, imagenet=is_imagenet)
elif model_name == 'res12_deltanet':
    # assert is_imagenet, 'Mainly for Imagenet'
    loginf("Model: Resnet12 + Deltanet")
    model = Res12DeltaModel(hidden_size=hidden_size, num_layers=num_layer,
                            num_head=n_head, dim_head=dim_head, dim_ff=dim_ff,
                            dropout=dropout_rate,
                            vision_dropout=vision_dropout,
                            num_classes=num_classes, imagenet=is_imagenet)
elif model_name == 'res12_srwm':
    # assert is_imagenet, 'Mainly for Imagenet'
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

# Set optimiser
learning_rate = args.learning_rate
clip = args.clip

############

best_model_path = os.path.join(args.load_from, 'best_model.pt')
lastest_model_path = os.path.join(args.load_from, 'lastest_model.pt')

loginf(f"[{datetime.now().strftime('%Y/%m/%d %H:%M:%S')}] Start Eval")

# load_from_checkpoint overwrite load_from
if args.load_from_checkpoint is not None:
    best_model_path = args.load_from_checkpoint

checkpoint = torch.load(best_model_path)
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()

results = []

num_test = args.num_test
test_size = args.test_size

for i in range(num_test):

    with torch.no_grad():
        test_total = eval_model_label_sync(
            model, test_dataloader, num_steps=args.test_size)

    test_total = 100 * test_total

    loginf(
        f"[test {i} {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}] "
        f'test total {test_total :.2f} %')

    results.append(test_total)

mean = np.mean(results)
std = np.std(results)

loginf(
    f'[{num_test} tests using {batch_size * test_size} samples each] '
    f'mean: {mean:.2f}, std: {std:.2f}, 95%-CI {1.96 * std / num_test:.2f}')

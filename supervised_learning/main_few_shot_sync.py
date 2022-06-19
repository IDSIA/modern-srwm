# Main file to be executed to train models for few shot learning in the
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
import torch.nn as nn
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
parser.add_argument('--fixed_valid', action='store_true',
                    help='use fixed validation set.')
parser.add_argument('--fixed_test', action='store_true',
                    help='use fixed test set.')
parser.add_argument('--total_epoch', default=1, type=int,
                    help='iterate more than one epoch.')
parser.add_argument('--train_acc_stop', default=120, type=int,
                    help='stopping based on train acc.')

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
parser.add_argument('--use_big_res12', action='store_true',
                    help='use big Res-12.')
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
                    help='param for torchmeta; number of query examples')

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
                         f"noshuf{args.disable_eval_shuffling}/" \
                         f"{args.test_per_class}-test_per_cl/" \
                         f"{args.n_way}way-{args.k_shot}shot/" \
                         f"L{args.num_layer}/h{args.hidden_size}/" \
                         f"n{args.n_head}/ff{args.ff_factor}/" \
                         f"d{args.dropout}/vd{args.vision_dropout}/" \
                         f"bigres{args.use_big_res12}/b{args.batch_size}/" \
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
    config.use_big_res12 = args.use_big_res12
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

dataset = data_cls(args.data_dir, ways=n_way, shots=k_shot_train,
                   test_shots=test_per_class, meta_train=True,
                   download=True, shuffle=True, seed=seed)
dataloader = BatchMetaDataLoader(
    dataset, batch_size=batch_size, num_workers=args.num_worker,
    pin_memory=True)

val_dataset = data_cls(args.data_dir, ways=n_way, shots=k_shot_train,
                       test_shots=test_per_class, meta_val=True,
                       shuffle=shuffled_eval, seed=valid_seed)

# this does not completely fix the valid set as the order of example is still
# randomized.
if args.fixed_valid:
    # https://github.com/tristandeleu/pytorch-meta/issues/132
    valid_class_size = len(val_dataset.dataset)  # num classes in valid
    # `dataset` here is torchmeta ClassDataset
    import itertools
    from torch.utils.data import Subset
    cls_indices = np.array(range(valid_class_size))
    all_indices = []
    for subset in itertools.combinations(cls_indices, args.n_way):
        all_indices.append(subset)
    val_total_size = args.valid_size * batch_size
    val_indices = random.sample(all_indices, val_total_size)
    val_dataset = Subset(val_dataset, val_indices)

val_dataloader = BatchMetaDataLoader(
    val_dataset, batch_size=batch_size, num_workers=args.num_worker,
    pin_memory=True)

test_dataset = data_cls(args.data_dir, ways=n_way, shots=k_shot_train,
                        test_shots=test_per_class, meta_test=True,
                        download=True, shuffle=shuffled_eval, seed=test_seed)

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
                           use_big=args.use_big_res12,
                           num_classes=num_classes, imagenet=is_imagenet)
elif model_name == 'res12_deltanet':
    # assert is_imagenet, 'Mainly for Imagenet'
    loginf("Model: Resnet12 + Deltanet")
    model = Res12DeltaModel(hidden_size=hidden_size, num_layers=num_layer,
                            num_head=n_head, dim_head=dim_head, dim_ff=dim_ff,
                            dropout=dropout_rate,
                            vision_dropout=vision_dropout,
                            use_big=args.use_big_res12,
                            num_classes=num_classes, imagenet=is_imagenet)
elif model_name == 'res12_srwm':
    # assert is_imagenet, 'Mainly for Imagenet'
    loginf("Model: Resnet12 + SRWM")
    model = Res12SRWMModel(hidden_size=hidden_size, num_layers=num_layer,
                           num_head=n_head, dim_head=dim_head, dim_ff=dim_ff,
                           dropout=dropout_rate, num_classes=num_classes,
                           vision_dropout=vision_dropout,
                           use_big=args.use_big_res12,
                           use_ln=True, beta_init=args.srwm_beta_init,
                           use_input_softmax=args.use_input_softmax,
                           imagenet=is_imagenet)

loginf(f"Number of trainable params: {model.num_params()}")
loginf(f"{model}")

model = model.to(device)

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

best_total_val_acc = 0.0

num_seq = 0
running_loss = 0.0
running_total = 0
running_correct = 0
run_step = 0

offset_step = 0
end_training = False
cur_train_acc = 0

for ep in range(args.total_epoch):
    loginf(f'epoch {ep} ====================')
    for i, batch in enumerate(dataloader):
        model.train()
        state = None

        train_inputs, train_targets = batch['train']
        train_inputs = train_inputs.to(device=device)  # (B, len, 1, 28, 28)
        train_targets = train_targets.to(device=device)  # (B, len)

        # shuffle and reshape
        train_shape = train_inputs.shape
        bsz, slen = train_shape[0], train_shape[1]

        num_seq += bsz

        train_inputs = train_inputs.transpose(0, 1)  # (len, B, 28 * 28)
        train_targets = train_targets.transpose(0, 1)  # (len, B)

        # same for test (i.e. query example)
        test_inputs, test_targets = batch['test']
        test_inputs = test_inputs.to(device=device)  # (B, test_len, 28 * 28)
        test_targets = test_targets.to(device=device)

        test_inputs = test_inputs.transpose(0, 1)  # (test_len, B, 28 * 28)
        test_targets = test_targets.transpose(0, 1)  # (test_len, B)

        # take only the fist element (randomized already)
        test_inputs = test_inputs[0].unsqueeze(0)
        test_targets = test_targets[0].unsqueeze(0)

        net_input = torch.cat([train_inputs, test_inputs], dim=0)
        target_labels = torch.cat([train_targets, test_targets], dim=0)

        target_labels_shape = target_labels.shape
        assert target_labels_shape[0] == slen + 1
        assert target_labels_shape[1] == bsz

        sync_labels = target_labels[:-1]
        # does not matter which label to feed for the last position as long
        # as it's consistently the same one
        dummy_last_token = torch.zeros_like(sync_labels[0].unsqueeze(0))
        label_feedback = torch.cat([sync_labels, dummy_last_token], dim=0)

        outputs, _ = model(net_input, label_feedback)
        outputs = outputs[-1]
        outputs = outputs.reshape(-1, num_classes)

        target_labels = target_labels[-1].reshape(-1)
        loss = loss_fn(outputs, target_labels)
        loss.backward()

        if i % args.grad_cummulate == 0:
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            model.reset_grad()

        # global loss
        running_loss += loss.item()
        running_total += target_labels.size(0)
        model.eval()
        with torch.no_grad():
            _, predicted = outputs.max(-1)
        bool_correct_pred = (predicted == target_labels)
        running_correct += bool_correct_pred.sum().item()

        run_step += 1
        if i % args.report_every == 0:
            cur_train_acc = 100 * running_correct / running_total
            if use_wandb:
                wandb.log({
                    "train_loss": running_loss / run_step,
                    "running_acc": 100 * running_correct / running_total,
                })

            train_elapsed = time.time() - train_timer
            train_timer = time.time()
            num_images_per_sec = (
                (i + 1 - last_batch_logged) * batch_size * (slen + 1)
                // train_elapsed)
            last_batch_logged = i

            loginf(
                f'steps: {i + offset_step}, num_seq: {num_seq}, '
                f'train_loss: {running_loss / run_step :.3f}, '
                f'running_acc: {100 * running_correct / running_total:.2f} % '
                f'(elapsed {int(train_elapsed)}s, {int(num_images_per_sec)} '
                'images/s)')
            running_loss = 0.0
            running_total = 0
            running_correct = 0
            run_step = 0

        if i % args.validate_every == 0:  # run validation
            model.eval()

            with torch.no_grad():
                v_total = eval_model_label_sync(
                    model, val_dataloader, num_steps=args.valid_size)

            loginf(
                f"[val {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}] "
                f'val total {100 * v_total :.2f} %, ')

            if use_wandb:
                wandb.log({
                    "val_acc": 100 * v_total,
                })

            if v_total > best_total_val_acc:
                best_total_val_acc = v_total
                best_step = i + offset_step
                # Save the best model
                loginf("The best model so far.")
                torch.save({'epoch': best_step,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'valid_acc': v_total}, best_model_path)
                loginf("Saved.")
            # Save the latest model
            torch.save({'train_step': i + offset_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'valid_total_acc': v_total}, lastest_model_path)

            elapsed = time.time() - interval_start_time
            loginf(f"Elapsed {elapsed / 60.:.2f} min since last valid.")
            interval_start_time = time.time()
            train_timer = time.time()
        if cur_train_acc > args.train_acc_stop:
            loginf(f'reached {args.train_acc_stop:.1f} % train accuracy')
            end_training = True
            break
        if i + offset_step > args.total_train_steps:
            end_training = True
            loginf(f'reached {args.total_train_steps} steps')
            break
    if end_training:
        break
    offset_step += i

elapsed = time.time() - start_time
loginf(f"Finished {i} steps in {elapsed / 60.:.2f} min.")
loginf(f"Best one shot validation acc: {100 * best_total_val_acc:.2f} % "
       f"at step {best_step}")

# load the best model and evaluate on the test set
del dataloader, dataset, val_dataloader, val_dataset

checkpoint = torch.load(best_model_path)
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()

with torch.no_grad():
    test_total = eval_model_label_sync(
        model, test_dataloader, num_steps=args.test_size)

loginf(
    f"[test {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}] "
    f'test total {100 * test_total :.2f} %')

# eval latest
checkpoint = torch.load(lastest_model_path)
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()

with torch.no_grad():
    test_total = eval_model_label_sync(
        model, test_dataloader, num_steps=args.test_size)

loginf(
    f"[test latest {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}] "
    f'test total {100 * test_total :.2f} %')

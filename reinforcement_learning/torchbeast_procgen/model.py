import nest

import torch
from torch import nn
from torch.nn import functional as F

from torchbeast.layer import DeltaNetLayer
from torchbeast.layer import LinearTransformerLayer
from torchbeast.layer import FastFFRecUpdateTanhLayer
from torchbeast.layer import FastRNNModelLayer
from torchbeast.layer import DeltaDeltaNetLayer
from torchbeast.layer import SRNetLayer, SMFWPNetLayer, NoCarryOverSRNetLayer, DeeperNetLayer, PseudoSRNetLayer


# Baseline model from torchbeast
class Net(nn.Module):
    def __init__(self, num_actions, conv_scale=1, use_lstm=False):
        super(Net, self).__init__()
        self.num_actions = num_actions
        self.use_lstm = use_lstm

        self.feat_convs = []
        self.resnet1 = []
        self.resnet2 = []

        self.convs = []

        input_channels = 3
        base_num_ch = [16, 32, 32]
        scaled_num_ch = [c * conv_scale for c in base_num_ch]
        for num_ch in scaled_num_ch:
            feats_convs = []
            feats_convs.append(
                nn.Conv2d(
                    in_channels=input_channels,
                    out_channels=num_ch,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            feats_convs.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            self.feat_convs.append(nn.Sequential(*feats_convs))

            input_channels = num_ch

            for i in range(2):
                resnet_block = []
                resnet_block.append(nn.ReLU())
                resnet_block.append(
                    nn.Conv2d(
                        in_channels=input_channels,
                        out_channels=num_ch,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )
                resnet_block.append(nn.ReLU())
                resnet_block.append(
                    nn.Conv2d(
                        in_channels=input_channels,
                        out_channels=num_ch,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )
                if i == 0:
                    self.resnet1.append(nn.Sequential(*resnet_block))
                else:
                    self.resnet2.append(nn.Sequential(*resnet_block))

        self.feat_convs = nn.ModuleList(self.feat_convs)
        self.resnet1 = nn.ModuleList(self.resnet1)
        self.resnet2 = nn.ModuleList(self.resnet2)

        self.fc = nn.Linear(2048 * conv_scale, 256)

        # FC output size + last reward.
        core_output_size = self.fc.out_features + 1

        if use_lstm:
            self.core = nn.LSTM(core_output_size, 256, num_layers=1)
            core_output_size = 256

        self.policy = nn.Linear(core_output_size, self.num_actions)
        self.baseline = nn.Linear(core_output_size, 1)

    def initial_state(self, batch_size=1):
        if not self.use_lstm:
            return tuple()
        return tuple(
            torch.zeros(self.core.num_layers, batch_size, self.core.hidden_size)
            for _ in range(2)
        )

    def forward(self, inputs, core_state):
        x = inputs["frame"]

        T, B, *_ = x.shape
        x = torch.flatten(x, 0, 1)  # Merge time and batch.
        x = x.float() / 255.0

        res_input = None
        for i, fconv in enumerate(self.feat_convs):
            x = fconv(x)
            res_input = x
            x = self.resnet1[i](x)
            x += res_input
            res_input = x
            x = self.resnet2[i](x)
            x += res_input

        x = F.relu(x)
        x = x.view(T * B, -1)
        x = F.relu(self.fc(x))

        clipped_reward = torch.clamp(inputs["reward"], -1, 1).view(T * B, 1)
        core_input = torch.cat([x, clipped_reward], dim=-1)

        if self.use_lstm:
            core_input = core_input.view(T, B, -1)
            core_output_list = []
            notdone = (~inputs["done"]).float()
            for input, nd in zip(core_input.unbind(), notdone.unbind()):
                # Reset core state to zero whenever an episode ended.
                # Make `done` broadcastable with (num_layers, B, hidden_size)
                # states:
                nd = nd.view(1, -1, 1)
                core_state = nest.map(nd.mul, core_state)
                output, core_state = self.core(input.unsqueeze(0), core_state)
                core_output_list.append(output)
            core_output = torch.flatten(torch.cat(core_output_list), 0, 1)
        else:
            core_output = core_input

        policy_logits = self.policy(core_output)
        baseline = self.baseline(core_output)

        if self.training:
            action = torch.multinomial(
                F.softmax(policy_logits, dim=1), num_samples=1)
        else:
            # Don't sample when testing.
            action = torch.argmax(policy_logits, dim=1)

        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)

        return (action, policy_logits, baseline), core_state


# Baseline model from torchbeast + Transformer FF layers
class DeeperNet(nn.Module):
    def __init__(self, num_actions, hidden_size, num_layers, dim_ff, dropout, use_lstm=False):
        super(DeeperNet, self).__init__()
        self.num_actions = num_actions
        self.use_lstm = use_lstm

        self.feat_convs = []
        self.resnet1 = []
        self.resnet2 = []

        self.convs = []

        input_channels = 3
        for num_ch in [16, 32, 32]:
            feats_convs = []
            feats_convs.append(
                nn.Conv2d(
                    in_channels=input_channels,
                    out_channels=num_ch,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            feats_convs.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            self.feat_convs.append(nn.Sequential(*feats_convs))

            input_channels = num_ch

            for i in range(2):
                resnet_block = []
                resnet_block.append(nn.ReLU())
                resnet_block.append(
                    nn.Conv2d(
                        in_channels=input_channels,
                        out_channels=num_ch,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )
                resnet_block.append(nn.ReLU())
                resnet_block.append(
                    nn.Conv2d(
                        in_channels=input_channels,
                        out_channels=num_ch,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )
                if i == 0:
                    self.resnet1.append(nn.Sequential(*resnet_block))
                else:
                    self.resnet2.append(nn.Sequential(*resnet_block))

        self.feat_convs = nn.ModuleList(self.feat_convs)
        self.resnet1 = nn.ModuleList(self.resnet1)
        self.resnet2 = nn.ModuleList(self.resnet2)

        self.fc = nn.Linear(2048, 256)

        # FC output size + last reward.
        core_output_size = self.fc.out_features + 1

        self.trafo_ff_block = DeeperNetLayer(core_output_size, hidden_size, num_layers, dim_ff, dropout)
        core_output_size = hidden_size

        self.policy = nn.Linear(core_output_size, self.num_actions)
        self.baseline = nn.Linear(core_output_size, 1)

    def initial_state(self, batch_size=1):
        if not self.use_lstm:
            return tuple()
        return tuple(
            torch.zeros(self.core.num_layers, batch_size, self.core.hidden_size)
            for _ in range(2)
        )

    def forward(self, inputs, core_state):
        x = inputs["frame"]

        T, B, *_ = x.shape
        x = torch.flatten(x, 0, 1)  # Merge time and batch.
        x = x.float() / 255.0

        res_input = None
        for i, fconv in enumerate(self.feat_convs):
            x = fconv(x)
            res_input = x
            x = self.resnet1[i](x)
            x += res_input
            res_input = x
            x = self.resnet2[i](x)
            x += res_input

        x = F.relu(x)
        x = x.view(T * B, -1)
        x = F.relu(self.fc(x))

        clipped_reward = torch.clamp(inputs["reward"], -1, 1).view(T * B, 1)
        core_input = torch.cat([x, clipped_reward], dim=-1)

        core_output = self.trafo_ff_block(core_input)

        policy_logits = self.policy(core_output)
        baseline = self.baseline(core_output)

        if self.training:
            action = torch.multinomial(
                F.softmax(policy_logits, dim=1), num_samples=1)
        else:
            # Don't sample when testing.
            action = torch.argmax(policy_logits, dim=1)

        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)

        return (action, policy_logits, baseline), core_state


class DeltaNetModel(nn.Module):
    def __init__(self, num_actions, hidden_size=128, num_layers=2,
                 num_head=4, dim_head=32, dim_ff=512, dropout=0.0):
        super(DeltaNetModel, self).__init__()
        self.num_actions = num_actions  # output vocab size.

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_head = num_head
        self.dim_head = dim_head
        self.dim_ff = dim_ff
        self.dropout = dropout

        self.feat_convs = []
        self.resnet1 = []
        self.resnet2 = []

        self.convs = []

        input_channels = 3
        for num_ch in [16, 32, 32]:
            feats_convs = []
            feats_convs.append(
                nn.Conv2d(
                    in_channels=input_channels,
                    out_channels=num_ch,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            feats_convs.append(
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            self.feat_convs.append(nn.Sequential(*feats_convs))

            input_channels = num_ch

            for i in range(2):
                resnet_block = []
                resnet_block.append(nn.ReLU())
                resnet_block.append(
                    nn.Conv2d(
                        in_channels=input_channels,
                        out_channels=num_ch,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )
                resnet_block.append(nn.ReLU())
                resnet_block.append(
                    nn.Conv2d(
                        in_channels=input_channels,
                        out_channels=num_ch,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )
                if i == 0:
                    self.resnet1.append(nn.Sequential(*resnet_block))
                else:
                    self.resnet2.append(nn.Sequential(*resnet_block))

        self.feat_convs = nn.ModuleList(self.feat_convs)
        self.resnet1 = nn.ModuleList(self.resnet1)
        self.resnet2 = nn.ModuleList(self.resnet2)

        self.fc = nn.Linear(2048, 256)

        # FC output size + last reward.
        # core_output_size = self.fc.out_features + 1
        self.core = DeltaNetLayer(self.fc.out_features + 1,
            hidden_size, num_layers, num_head, dim_head, dim_ff, dropout)
        core_output_size = hidden_size

        self.policy = nn.Linear(core_output_size, self.num_actions)
        self.baseline = nn.Linear(core_output_size, 1)

    def initial_state(self, batch_size=1):
        state_tuple = tuple(torch.zeros(
            1, batch_size, self.num_head, self.dim_head, self.dim_head)
            for _ in range(self.num_layers))
        return state_tuple

    def forward(self, inputs, core_state):
        x = inputs["frame"]
        assert x.device is not 'cpu'
        T, B, *_ = x.shape
        x = torch.flatten(x, 0, 1)  # Merge time and batch.
        x = x.float() / 255.0

        res_input = None
        for i, fconv in enumerate(self.feat_convs):
            x = fconv(x)
            res_input = x
            x = self.resnet1[i](x)
            x += res_input
            res_input = x
            x = self.resnet2[i](x)
            x += res_input

        x = F.relu(x)
        x = x.view(T * B, -1)
        x = F.relu(self.fc(x))

        clipped_reward = torch.clamp(inputs["reward"], -1, 1).view(T * B, 1)
        core_input = torch.cat([x, clipped_reward], dim=-1)

        # recurrent component
        core_input = core_input.view(T, B, -1)
        core_output_list = []
        notdone = (~inputs["done"]).float()
        for input, nd in zip(core_input.unbind(), notdone.unbind()):
            # Reset core state to zero whenever an episode ended.
            # Make `done` broadcastable with (num_layers, B, D, D)
            # states:
            # nd = nd.view(1, -1, 1)
            # needs extra dim0 for compat w/ inference code
            nd = nd.view(1, -1, 1, 1, 1)
            core_state = nest.map(nd.mul, core_state)
            output, core_state = self.core(input.unsqueeze(0), core_state)
            core_output_list.append(output)
        core_output = torch.flatten(torch.cat(core_output_list), 0, 1)

        policy_logits = self.policy(core_output)
        baseline = self.baseline(core_output)

        if self.training:
            action = torch.multinomial(F.softmax(policy_logits, dim=1),
                                       num_samples=1)
        else:
            # Don't sample when testing.
            action = torch.argmax(policy_logits, dim=1)

        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)

        return (action, policy_logits, baseline), core_state


class LinearTransformerModel(nn.Module):
    def __init__(self, num_actions, hidden_size=128, num_layers=2,
                 num_head=4, dim_head=32, dim_ff=512, dropout=0.0):
        super(LinearTransformerModel, self).__init__()
        self.num_actions = num_actions  # output vocab size.

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_head = num_head
        self.dim_head = dim_head
        self.dim_ff = dim_ff
        self.dropout = dropout

        self.feat_convs = []
        self.resnet1 = []
        self.resnet2 = []

        self.convs = []

        input_channels = 3
        for num_ch in [16, 32, 32]:
            feats_convs = []
            feats_convs.append(
                nn.Conv2d(
                    in_channels=input_channels,
                    out_channels=num_ch,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            feats_convs.append(
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            self.feat_convs.append(nn.Sequential(*feats_convs))

            input_channels = num_ch

            for i in range(2):
                resnet_block = []
                resnet_block.append(nn.ReLU())
                resnet_block.append(
                    nn.Conv2d(
                        in_channels=input_channels,
                        out_channels=num_ch,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )
                resnet_block.append(nn.ReLU())
                resnet_block.append(
                    nn.Conv2d(
                        in_channels=input_channels,
                        out_channels=num_ch,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )
                if i == 0:
                    self.resnet1.append(nn.Sequential(*resnet_block))
                else:
                    self.resnet2.append(nn.Sequential(*resnet_block))

        self.feat_convs = nn.ModuleList(self.feat_convs)
        self.resnet1 = nn.ModuleList(self.resnet1)
        self.resnet2 = nn.ModuleList(self.resnet2)

        self.fc = nn.Linear(2048, 256)

        # FC output size + last reward.
        # core_output_size = self.fc.out_features + 1
        self.core = LinearTransformerLayer(self.fc.out_features + 1,
            hidden_size, num_layers, num_head, dim_head, dim_ff, dropout)
        core_output_size = hidden_size

        self.policy = nn.Linear(core_output_size, self.num_actions)
        self.baseline = nn.Linear(core_output_size, 1)

    def initial_state(self, batch_size=1):
        state_tuple = tuple(torch.zeros(
            1, batch_size, self.num_head, self.dim_head, self.dim_head)
            for _ in range(self.num_layers))
        return state_tuple

    def forward(self, inputs, core_state):
        x = inputs["frame"]
        assert x.device is not 'cpu'
        T, B, *_ = x.shape
        x = torch.flatten(x, 0, 1)  # Merge time and batch.
        x = x.float() / 255.0

        res_input = None
        for i, fconv in enumerate(self.feat_convs):
            x = fconv(x)
            res_input = x
            x = self.resnet1[i](x)
            x += res_input
            res_input = x
            x = self.resnet2[i](x)
            x += res_input

        x = F.relu(x)
        x = x.view(T * B, -1)
        x = F.relu(self.fc(x))

        clipped_reward = torch.clamp(inputs["reward"], -1, 1).view(T * B, 1)
        core_input = torch.cat([x, clipped_reward], dim=-1)

        # recurrent component
        core_input = core_input.view(T, B, -1)
        core_output_list = []
        notdone = (~inputs["done"]).float()
        for input, nd in zip(core_input.unbind(), notdone.unbind()):
            # Reset core state to zero whenever an episode ended.
            # Make `done` broadcastable with (num_layers, B, D, D)
            # states:
            # nd = nd.view(1, -1, 1)
            # needs extra dim0 for compat w/ inference code
            nd = nd.view(1, -1, 1, 1, 1)
            core_state = nest.map(nd.mul, core_state)
            output, core_state = self.core(input.unsqueeze(0), core_state)
            core_output_list.append(output)
        core_output = torch.flatten(torch.cat(core_output_list), 0, 1)

        policy_logits = self.policy(core_output)
        baseline = self.baseline(core_output)

        if self.training:
            action = torch.multinomial(F.softmax(policy_logits, dim=1),
                                       num_samples=1)
        else:
            # Don't sample when testing.
            action = torch.argmax(policy_logits, dim=1)

        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)

        return (action, policy_logits, baseline), core_state


class RecDeltaModel(nn.Module):
    def __init__(self, num_actions, hidden_size=128, num_layers=2,
                 num_head=4, dim_head=32, dim_ff=512, dropout=0.0):
        super(RecDeltaModel, self).__init__()
        self.num_actions = num_actions  # output vocab size.

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_head = num_head
        self.dim_head = dim_head
        self.dim_ff = dim_ff
        self.dropout = dropout

        self.feat_convs = []
        self.resnet1 = []
        self.resnet2 = []

        self.convs = []

        input_channels = 3
        for num_ch in [16, 32, 32]:
            feats_convs = []
            feats_convs.append(
                nn.Conv2d(
                    in_channels=input_channels,
                    out_channels=num_ch,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            feats_convs.append(
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            self.feat_convs.append(nn.Sequential(*feats_convs))

            input_channels = num_ch

            for i in range(2):
                resnet_block = []
                resnet_block.append(nn.ReLU())
                resnet_block.append(
                    nn.Conv2d(
                        in_channels=input_channels,
                        out_channels=num_ch,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )
                resnet_block.append(nn.ReLU())
                resnet_block.append(
                    nn.Conv2d(
                        in_channels=input_channels,
                        out_channels=num_ch,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )
                if i == 0:
                    self.resnet1.append(nn.Sequential(*resnet_block))
                else:
                    self.resnet2.append(nn.Sequential(*resnet_block))

        self.feat_convs = nn.ModuleList(self.feat_convs)
        self.resnet1 = nn.ModuleList(self.resnet1)
        self.resnet2 = nn.ModuleList(self.resnet2)

        self.fc = nn.Linear(2048, 256)

        # FC output size + last reward.
        # core_output_size = self.fc.out_features + 1
        self.core = FastFFRecUpdateTanhLayer(self.fc.out_features + 1,
            hidden_size, num_layers, num_head, dim_head, dim_ff, dropout)
        core_output_size = hidden_size

        self.policy = nn.Linear(core_output_size, self.num_actions)
        self.baseline = nn.Linear(core_output_size, 1)

    def initial_state(self, batch_size=1):
        # why not create on device?
        # add dummy dim0 for inference code compat.

        fw_state_tuple = tuple(
            torch.zeros(
                1, batch_size, self.num_head, self.dim_head, self.dim_head)
            for _ in range(self.num_layers)
        )

        rnn_state_tuple = tuple(
            torch.zeros(1, batch_size, self.num_head, 1, self.dim_head)
            for _ in range(self.num_layers)
        )
        return (fw_state_tuple, rnn_state_tuple)

    def forward(self, inputs, core_state):
        x = inputs["frame"]
        assert x.device is not 'cpu'
        T, B, *_ = x.shape
        x = torch.flatten(x, 0, 1)  # Merge time and batch.
        x = x.float() / 255.0

        res_input = None
        for i, fconv in enumerate(self.feat_convs):
            x = fconv(x)
            res_input = x
            x = self.resnet1[i](x)
            x += res_input
            res_input = x
            x = self.resnet2[i](x)
            x += res_input

        x = F.relu(x)
        x = x.view(T * B, -1)
        x = F.relu(self.fc(x))

        clipped_reward = torch.clamp(inputs["reward"], -1, 1).view(T * B, 1)
        core_input = torch.cat([x, clipped_reward], dim=-1)

        # recurrent component
        core_input = core_input.view(T, B, -1)
        core_output_list = []
        notdone = (~inputs["done"]).float()
        for input, nd in zip(core_input.unbind(), notdone.unbind()):
            # Reset core state to zero whenever an episode ended.
            # Make `done` broadcastable with (num_layers, B, D, D)
            # states:
            # nd = nd.view(1, -1, 1)
            # needs extra dim0 for compat w/ inference code
            fw_state, rnn_state = core_state
            nd = nd.view(1, -1, 1, 1, 1)
            fw_state = nest.map(nd.mul, fw_state)
            rnn_state = nest.map(nd.mul, rnn_state)
            core_state = (fw_state, rnn_state)
            output, core_state = self.core(input.unsqueeze(0), core_state)
            core_output_list.append(output)
        core_output = torch.flatten(torch.cat(core_output_list), 0, 1)

        policy_logits = self.policy(core_output)
        baseline = self.baseline(core_output)

        if self.training:
            action = torch.multinomial(F.softmax(policy_logits, dim=1),
                                       num_samples=1)
        else:
            # Don't sample when testing.
            action = torch.argmax(policy_logits, dim=1)

        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)

        return (action, policy_logits, baseline), core_state


class FastRNNModel(nn.Module):
    def __init__(self, num_actions, hidden_size=128, num_layers=2,
                 num_head=4, dim_head=32, dim_ff=512, dropout=0.0):
        super(FastRNNModel, self).__init__()
        self.num_actions = num_actions  # output vocab size.

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_head = num_head
        self.dim_head = dim_head
        self.dim_ff = dim_ff
        self.dropout = dropout

        self.feat_convs = []
        self.resnet1 = []
        self.resnet2 = []

        self.convs = []

        input_channels = 3
        for num_ch in [16, 32, 32]:
            feats_convs = []
            feats_convs.append(
                nn.Conv2d(
                    in_channels=input_channels,
                    out_channels=num_ch,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            feats_convs.append(
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            self.feat_convs.append(nn.Sequential(*feats_convs))

            input_channels = num_ch

            for i in range(2):
                resnet_block = []
                resnet_block.append(nn.ReLU())
                resnet_block.append(
                    nn.Conv2d(
                        in_channels=input_channels,
                        out_channels=num_ch,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )
                resnet_block.append(nn.ReLU())
                resnet_block.append(
                    nn.Conv2d(
                        in_channels=input_channels,
                        out_channels=num_ch,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )
                if i == 0:
                    self.resnet1.append(nn.Sequential(*resnet_block))
                else:
                    self.resnet2.append(nn.Sequential(*resnet_block))

        self.feat_convs = nn.ModuleList(self.feat_convs)
        self.resnet1 = nn.ModuleList(self.resnet1)
        self.resnet2 = nn.ModuleList(self.resnet2)

        self.fc = nn.Linear(2048, 256)

        # FC output size + last reward.
        # core_output_size = self.fc.out_features + 1
        self.core = FastRNNModelLayer(self.fc.out_features + 1,
            hidden_size, num_layers, num_head, dim_head, dim_ff, dropout)
        core_output_size = hidden_size

        self.policy = nn.Linear(core_output_size, self.num_actions)
        self.baseline = nn.Linear(core_output_size, 1)

    def initial_state(self, batch_size=1):

        fw_state_tuple = tuple(
            torch.zeros(
                1, batch_size, self.num_head, self.dim_head, self.dim_head)
            for _ in range(self.num_layers)
        )

        rec_fw_state_tuple = tuple(
            torch.zeros(
                1, batch_size, self.num_head, self.dim_head, self.dim_head)
            for _ in range(self.num_layers)
        )

        rnn_state_tuple = tuple(
            torch.zeros(1, batch_size, self.num_head, 1, self.dim_head)
            for _ in range(self.num_layers)
        )
        return (fw_state_tuple, rec_fw_state_tuple, rnn_state_tuple)

    def forward(self, inputs, core_state):
        x = inputs["frame"]
        assert x.device is not 'cpu'
        T, B, *_ = x.shape
        x = torch.flatten(x, 0, 1)  # Merge time and batch.
        x = x.float() / 255.0

        res_input = None
        for i, fconv in enumerate(self.feat_convs):
            x = fconv(x)
            res_input = x
            x = self.resnet1[i](x)
            x += res_input
            res_input = x
            x = self.resnet2[i](x)
            x += res_input

        x = F.relu(x)
        x = x.view(T * B, -1)
        x = F.relu(self.fc(x))

        clipped_reward = torch.clamp(inputs["reward"], -1, 1).view(T * B, 1)
        core_input = torch.cat([x, clipped_reward], dim=-1)

        # recurrent component
        core_input = core_input.view(T, B, -1)
        core_output_list = []
        notdone = (~inputs["done"]).float()
        for input, nd in zip(core_input.unbind(), notdone.unbind()):
            # Reset core state to zero whenever an episode ended.
            # Make `done` broadcastable with (num_layers, B, D, D)
            # states:
            # nd = nd.view(1, -1, 1)
            # needs extra dim0 for compat w/ inference code
            fw_state, rec_fw_state, rnn_state = core_state
            nd = nd.view(1, -1, 1, 1, 1)
            fw_state = nest.map(nd.mul, fw_state)
            rec_fw_state = nest.map(nd.mul, rec_fw_state)
            rnn_state = nest.map(nd.mul, rnn_state)
            core_state = (fw_state, rec_fw_state, rnn_state)
            output, core_state = self.core(input.unsqueeze(0), core_state)
            core_output_list.append(output)
        core_output = torch.flatten(torch.cat(core_output_list), 0, 1)

        policy_logits = self.policy(core_output)
        baseline = self.baseline(core_output)

        if self.training:
            action = torch.multinomial(F.softmax(policy_logits, dim=1),
                                       num_samples=1)
        else:
            # Don't sample when testing.
            action = torch.argmax(policy_logits, dim=1)

        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)

        return (action, policy_logits, baseline), core_state


class DeltaDeltaNetModel(nn.Module):
    def __init__(self, num_actions, hidden_size=128, num_layers=2,
                 num_head=4, dim_head=32, dim_ff=512, dropout=0.0,
                 use_xem=False):
        # use_xem: use cross episode memory
        super(DeltaDeltaNetModel, self).__init__()
        self.num_actions = num_actions  # output vocab size.
        self.use_xem = use_xem

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_head = num_head
        self.dim_head = dim_head
        self.dim_ff = dim_ff
        self.dropout = dropout

        self.feat_convs = []
        self.resnet1 = []
        self.resnet2 = []

        self.convs = []

        input_channels = 3
        for num_ch in [16, 32, 32]:
            feats_convs = []
            feats_convs.append(
                nn.Conv2d(
                    in_channels=input_channels,
                    out_channels=num_ch,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            feats_convs.append(
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            self.feat_convs.append(nn.Sequential(*feats_convs))

            input_channels = num_ch

            for i in range(2):
                resnet_block = []
                resnet_block.append(nn.ReLU())
                resnet_block.append(
                    nn.Conv2d(
                        in_channels=input_channels,
                        out_channels=num_ch,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )
                resnet_block.append(nn.ReLU())
                resnet_block.append(
                    nn.Conv2d(
                        in_channels=input_channels,
                        out_channels=num_ch,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )
                if i == 0:
                    self.resnet1.append(nn.Sequential(*resnet_block))
                else:
                    self.resnet2.append(nn.Sequential(*resnet_block))

        self.feat_convs = nn.ModuleList(self.feat_convs)
        self.resnet1 = nn.ModuleList(self.resnet1)
        self.resnet2 = nn.ModuleList(self.resnet2)

        self.fc = nn.Linear(2048, 256)

        # FC output size + last reward.
        # core_output_size = self.fc.out_features + 1
        self.core = DeltaDeltaNetLayer(
            self.fc.out_features + 1, hidden_size, num_layers, num_head,
            dim_head, dim_ff, dropout)
        core_output_size = hidden_size

        self.policy = nn.Linear(core_output_size, self.num_actions)
        self.baseline = nn.Linear(core_output_size, 1)

    def initial_state(self, batch_size=1):
        fw_state_tuple = tuple(
            torch.zeros(
                1, batch_size, self.num_head, self.dim_head,
                3 * self.dim_head + 1)
            for _ in range(self.num_layers)
        )

        very_fw_state_tuple = tuple(
            torch.zeros(
                1, batch_size, self.num_head, self.dim_head, self.dim_head)
            for _ in range(self.num_layers)
        )

        return (fw_state_tuple, very_fw_state_tuple)

    def forward(self, inputs, core_state):
        x = inputs["frame"]
        assert x.device is not 'cpu'
        T, B, *_ = x.shape
        x = torch.flatten(x, 0, 1)  # Merge time and batch.
        x = x.float() / 255.0

        res_input = None
        for i, fconv in enumerate(self.feat_convs):
            x = fconv(x)
            res_input = x
            x = self.resnet1[i](x)
            x += res_input
            res_input = x
            x = self.resnet2[i](x)
            x += res_input

        x = F.relu(x)
        x = x.view(T * B, -1)
        x = F.relu(self.fc(x))

        clipped_reward = torch.clamp(inputs["reward"], -1, 1).view(T * B, 1)
        core_input = torch.cat([x, clipped_reward], dim=-1)

        # recurrent component
        core_input = core_input.view(T, B, -1)
        core_output_list = []
        notdone = (~inputs["done"]).float()
        for input, nd in zip(core_input.unbind(), notdone.unbind()):
            # if not use_xem
            # Reset core state to zero whenever an episode ended.
            # Make `done` broadcastable with (num_layers, B, D, D)
            # states:
            # nd = nd.view(1, -1, 1)
            # needs extra dim0 for compat w/ inference code
            fw_state, very_fw_state = core_state
            nd = nd.view(1, -1, 1, 1, 1)
            if not self.use_xem:
                fw_state = nest.map(nd.mul, fw_state)
            else:
                # save cross episodic fast weights
                layer_id = 0
                for fw_layer in self.core.fwm_layers:
                    fw_layer.cached_fast_weights = fw_state[layer_id][0]
                    layer_id += 1

            very_fw_state = nest.map(nd.mul, very_fw_state)
            core_state = (fw_state, very_fw_state)
            output, core_state = self.core(input.unsqueeze(0), core_state)
            core_output_list.append(output)

        core_output = torch.flatten(torch.cat(core_output_list), 0, 1)

        policy_logits = self.policy(core_output)
        baseline = self.baseline(core_output)

        if self.training:
            action = torch.multinomial(F.softmax(policy_logits, dim=1),
                                       num_samples=1)
        else:
            # Don't sample when testing.
            action = torch.argmax(policy_logits, dim=1)

        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)

        return (action, policy_logits, baseline), core_state


# Outer product based self-referential matrix
class SRModel(nn.Module):
    def __init__(self, num_actions, hidden_size=128, num_layers=2,
                 num_head=4, dim_head=16, dim_ff=512, dropout=0.0,
                 use_xem=False):
        super(SRModel, self).__init__()
        self.num_actions = num_actions  # output vocab size.
        self.use_xem = use_xem

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_head = num_head
        self.dim_head = dim_head
        self.dim_ff = dim_ff
        self.dropout = dropout

        self.feat_convs = []
        self.resnet1 = []
        self.resnet2 = []

        self.convs = []

        input_channels = 3
        for num_ch in [16, 32, 32]:
            feats_convs = []
            feats_convs.append(
                nn.Conv2d(
                    in_channels=input_channels,
                    out_channels=num_ch,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            feats_convs.append(
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            self.feat_convs.append(nn.Sequential(*feats_convs))

            input_channels = num_ch

            for i in range(2):
                resnet_block = []
                resnet_block.append(nn.ReLU())
                resnet_block.append(
                    nn.Conv2d(
                        in_channels=input_channels,
                        out_channels=num_ch,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )
                resnet_block.append(nn.ReLU())
                resnet_block.append(
                    nn.Conv2d(
                        in_channels=input_channels,
                        out_channels=num_ch,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )
                if i == 0:
                    self.resnet1.append(nn.Sequential(*resnet_block))
                else:
                    self.resnet2.append(nn.Sequential(*resnet_block))

        self.feat_convs = nn.ModuleList(self.feat_convs)
        self.resnet1 = nn.ModuleList(self.resnet1)
        self.resnet2 = nn.ModuleList(self.resnet2)

        self.fc = nn.Linear(2048, 256)

        # FC output size + last reward.
        # core_output_size = self.fc.out_features + 1
        self.core = SRNetLayer(self.fc.out_features + 1,
            hidden_size, num_layers, num_head, dim_head, dim_ff, dropout)
        core_output_size = hidden_size

        self.policy = nn.Linear(core_output_size, self.num_actions)
        self.baseline = nn.Linear(core_output_size, 1)

    def initial_state(self, bsz=1):

        Wy_state_tuple = tuple(
            torch.zeros(1, bsz, self.num_head, self.dim_head, self.dim_head)
            for _ in range(self.num_layers)
        )

        Wq_state_tuple = tuple(
            torch.zeros(1, bsz, self.num_head, self.dim_head, self.dim_head)
            for _ in range(self.num_layers)
        )

        Wk_state_tuple = tuple(
            torch.zeros(1, bsz, self.num_head, self.dim_head, self.dim_head)
            for _ in range(self.num_layers)
        )

        wb_state_tuple = tuple(
            torch.zeros(1, bsz, self.num_head, self.dim_head, 4)
            for _ in range(self.num_layers)
        )

        return (Wy_state_tuple, Wq_state_tuple, Wk_state_tuple, wb_state_tuple)

    def forward(self, inputs, core_state):
        x = inputs["frame"]
        assert x.device is not 'cpu'
        T, B, *_ = x.shape
        x = torch.flatten(x, 0, 1)  # Merge time and batch.
        x = x.float() / 255.0

        res_input = None
        for i, fconv in enumerate(self.feat_convs):
            x = fconv(x)
            res_input = x
            x = self.resnet1[i](x)
            x += res_input
            res_input = x
            x = self.resnet2[i](x)
            x += res_input

        x = F.relu(x)
        x = x.view(T * B, -1)
        x = F.relu(self.fc(x))

        clipped_reward = torch.clamp(inputs["reward"], -1, 1).view(T * B, 1)
        core_input = torch.cat([x, clipped_reward], dim=-1)

        # recurrent component
        core_input = core_input.view(T, B, -1)
        core_output_list = []
        notdone = (~inputs["done"]).float()
        for input, nd in zip(core_input.unbind(), notdone.unbind()):
            # Reset core state to zero whenever an episode ended.
            # Make `done` broadcastable with (num_layers, B, D, D)
            # states:
            # nd = nd.view(1, -1, 1)
            # needs extra dim0 for compat w/ inference code
            nd = nd.view(1, -1, 1, 1, 1)
            # better reset?
            (Wy_s, Wq_s, Wk_s, wb_s) = core_state
            if not self.use_xem:
                Wy_s = nest.map(nd.mul, Wy_s)
                Wq_s = nest.map(nd.mul, Wq_s)
                Wk_s = nest.map(nd.mul, Wk_s)
                wb_s = nest.map(nd.mul, wb_s)
            core_state = (Wy_s, Wq_s, Wk_s, wb_s)

            output, core_state = self.core(input.unsqueeze(0), core_state)
            core_output_list.append(output)

        core_output = torch.flatten(torch.cat(core_output_list), 0, 1)

        policy_logits = self.policy(core_output)
        baseline = self.baseline(core_output)

        if self.training:
            action = torch.multinomial(F.softmax(policy_logits, dim=1),
                                       num_samples=1)
        else:
            # Don't sample when testing.
            action = torch.argmax(policy_logits, dim=1)

        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)

        return (action, policy_logits, baseline), core_state


# 'Fake SR' in the paper
class PseudoSRModel(nn.Module):
    def __init__(self, num_actions, hidden_size=128, num_layers=2,
                 num_head=4, dim_head=16, dim_ff=512, dropout=0.0,
                 use_xem=False):
        super(PseudoSRModel, self).__init__()
        self.num_actions = num_actions  # output vocab size.
        self.use_xem = use_xem

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_head = num_head
        self.dim_head = dim_head
        self.dim_ff = dim_ff
        self.dropout = dropout

        self.feat_convs = []
        self.resnet1 = []
        self.resnet2 = []

        self.convs = []

        input_channels = 3
        for num_ch in [16, 32, 32]:
            feats_convs = []
            feats_convs.append(
                nn.Conv2d(
                    in_channels=input_channels,
                    out_channels=num_ch,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            feats_convs.append(
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            self.feat_convs.append(nn.Sequential(*feats_convs))

            input_channels = num_ch

            for i in range(2):
                resnet_block = []
                resnet_block.append(nn.ReLU())
                resnet_block.append(
                    nn.Conv2d(
                        in_channels=input_channels,
                        out_channels=num_ch,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )
                resnet_block.append(nn.ReLU())
                resnet_block.append(
                    nn.Conv2d(
                        in_channels=input_channels,
                        out_channels=num_ch,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )
                if i == 0:
                    self.resnet1.append(nn.Sequential(*resnet_block))
                else:
                    self.resnet2.append(nn.Sequential(*resnet_block))

        self.feat_convs = nn.ModuleList(self.feat_convs)
        self.resnet1 = nn.ModuleList(self.resnet1)
        self.resnet2 = nn.ModuleList(self.resnet2)

        self.fc = nn.Linear(2048, 256)

        # FC output size + last reward.
        # core_output_size = self.fc.out_features + 1
        self.core = PseudoSRNetLayer(self.fc.out_features + 1,
            hidden_size, num_layers, num_head, dim_head, dim_ff, dropout)
        core_output_size = hidden_size

        self.policy = nn.Linear(core_output_size, self.num_actions)
        self.baseline = nn.Linear(core_output_size, 1)

    def initial_state(self, batch_size=1):
        return tuple()

    def forward(self, inputs, core_state):
        x = inputs["frame"]
        assert x.device is not 'cpu'
        T, B, *_ = x.shape
        x = torch.flatten(x, 0, 1)  # Merge time and batch.
        x = x.float() / 255.0

        res_input = None
        for i, fconv in enumerate(self.feat_convs):
            x = fconv(x)
            res_input = x
            x = self.resnet1[i](x)
            x += res_input
            res_input = x
            x = self.resnet2[i](x)
            x += res_input

        x = F.relu(x)
        x = x.view(T * B, -1)
        x = F.relu(self.fc(x))

        clipped_reward = torch.clamp(inputs["reward"], -1, 1).view(T * B, 1)
        core_input = torch.cat([x, clipped_reward], dim=-1)

        # recurrent component
        core_input = core_input.view(T, B, -1)
        core_output_list = []
        notdone = (~inputs["done"]).float()

        core_output, _ = self.core(core_input, core_state=None)
        core_output = torch.flatten(core_output, 0, 1)

        policy_logits = self.policy(core_output)
        baseline = self.baseline(core_output)

        if self.training:
            action = torch.multinomial(F.softmax(policy_logits, dim=1),
                                       num_samples=1)
        else:
            # Don't sample when testing.
            action = torch.argmax(policy_logits, dim=1)

        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)

        return (action, policy_logits, baseline), core_state


# Outer product based self-referential matrix, w/o carry over contexts
class NoCarryOverSRModel(nn.Module):
    def __init__(self, num_actions, hidden_size=128, num_layers=2,
                 num_head=4, dim_head=16, dim_ff=512, dropout=0.0):
        super(NoCarryOverSRModel, self).__init__()
        self.num_actions = num_actions  # output vocab size.

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_head = num_head
        self.dim_head = dim_head
        self.dim_ff = dim_ff
        self.dropout = dropout

        self.feat_convs = []
        self.resnet1 = []
        self.resnet2 = []

        self.convs = []

        input_channels = 3
        for num_ch in [16, 32, 32]:
            feats_convs = []
            feats_convs.append(
                nn.Conv2d(
                    in_channels=input_channels,
                    out_channels=num_ch,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            feats_convs.append(
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            self.feat_convs.append(nn.Sequential(*feats_convs))

            input_channels = num_ch

            for i in range(2):
                resnet_block = []
                resnet_block.append(nn.ReLU())
                resnet_block.append(
                    nn.Conv2d(
                        in_channels=input_channels,
                        out_channels=num_ch,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )
                resnet_block.append(nn.ReLU())
                resnet_block.append(
                    nn.Conv2d(
                        in_channels=input_channels,
                        out_channels=num_ch,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )
                if i == 0:
                    self.resnet1.append(nn.Sequential(*resnet_block))
                else:
                    self.resnet2.append(nn.Sequential(*resnet_block))

        self.feat_convs = nn.ModuleList(self.feat_convs)
        self.resnet1 = nn.ModuleList(self.resnet1)
        self.resnet2 = nn.ModuleList(self.resnet2)

        self.fc = nn.Linear(2048, 256)

        # FC output size + last reward.
        # core_output_size = self.fc.out_features + 1
        self.core = NoCarryOverSRNetLayer(self.fc.out_features + 1,
            hidden_size, num_layers, num_head, dim_head, dim_ff, dropout)
        core_output_size = hidden_size

        self.policy = nn.Linear(core_output_size, self.num_actions)
        self.baseline = nn.Linear(core_output_size, 1)

    def initial_state(self, bsz=1):
        return tuple()

    def forward(self, inputs, core_state):
        x = inputs["frame"]
        assert x.device is not 'cpu'
        T, B, *_ = x.shape
        x = torch.flatten(x, 0, 1)  # Merge time and batch.
        x = x.float() / 255.0

        res_input = None
        for i, fconv in enumerate(self.feat_convs):
            x = fconv(x)
            res_input = x
            x = self.resnet1[i](x)
            x += res_input
            res_input = x
            x = self.resnet2[i](x)
            x += res_input

        x = F.relu(x)
        x = x.view(T * B, -1)
        x = F.relu(self.fc(x))

        clipped_reward = torch.clamp(inputs["reward"], -1, 1).view(T * B, 1)
        core_input = torch.cat([x, clipped_reward], dim=-1)

        # recurrent component
        core_input = core_input.view(T, B, -1)
        core_output_list = []
        notdone = (~inputs["done"]).float()
        for input, nd in zip(core_input.unbind(), notdone.unbind()):
            # Reset core state to zero whenever an episode ended.
            # Make `done` broadcastable with (num_layers, B, D, D)
            # states:
            # nd = nd.view(1, -1, 1)
            # needs extra dim0 for compat w/ inference code
            # nd = nd.view(1, -1, 1, 1, 1)

            output, core_state = self.core(input.unsqueeze(0), core_state)
            core_output_list.append(output)

        core_output = torch.flatten(torch.cat(core_output_list), 0, 1)

        policy_logits = self.policy(core_output)
        baseline = self.baseline(core_output)

        if self.training:
            action = torch.multinomial(F.softmax(policy_logits, dim=1),
                                       num_samples=1)
        else:
            # Don't sample when testing.
            action = torch.argmax(policy_logits, dim=1)

        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)

        return (action, policy_logits, baseline), core_state


# Self-modifying FWP model, 'SR-Delta'
class SMFWPModel(nn.Module):
    def __init__(self, num_actions, hidden_size=128, num_layers=2,
                 num_head=4, dim_head=16, dim_ff=512, dropout=0.0):
        super(SMFWPModel, self).__init__()
        self.num_actions = num_actions  # output vocab size.

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_head = num_head
        self.dim_head = dim_head
        self.dim_ff = dim_ff
        self.dropout = dropout
        self.y_d_head = 3 * dim_head + 1

        self.feat_convs = []
        self.resnet1 = []
        self.resnet2 = []

        self.convs = []

        input_channels = 3
        for num_ch in [16, 32, 32]:
            feats_convs = []
            feats_convs.append(
                nn.Conv2d(
                    in_channels=input_channels,
                    out_channels=num_ch,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            feats_convs.append(
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            self.feat_convs.append(nn.Sequential(*feats_convs))

            input_channels = num_ch

            for i in range(2):
                resnet_block = []
                resnet_block.append(nn.ReLU())
                resnet_block.append(
                    nn.Conv2d(
                        in_channels=input_channels,
                        out_channels=num_ch,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )
                resnet_block.append(nn.ReLU())
                resnet_block.append(
                    nn.Conv2d(
                        in_channels=input_channels,
                        out_channels=num_ch,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )
                if i == 0:
                    self.resnet1.append(nn.Sequential(*resnet_block))
                else:
                    self.resnet2.append(nn.Sequential(*resnet_block))

        self.feat_convs = nn.ModuleList(self.feat_convs)
        self.resnet1 = nn.ModuleList(self.resnet1)
        self.resnet2 = nn.ModuleList(self.resnet2)

        self.fc = nn.Linear(2048, 256)

        # FC output size + last reward.
        # core_output_size = self.fc.out_features + 1
        self.core = SMFWPNetLayer(self.fc.out_features + 1,
            hidden_size, num_layers, num_head, dim_head, dim_ff, dropout)
        core_output_size = hidden_size

        self.policy = nn.Linear(core_output_size, self.num_actions)
        self.baseline = nn.Linear(core_output_size, 1)

    def initial_state(self, bsz=1):

        Wy_state_tuple = tuple(
            torch.zeros(1, bsz, self.num_head, self.dim_head, self.y_d_head)
            for _ in range(self.num_layers)
        )

        Wq_state_tuple = tuple(
            torch.zeros(1, bsz, self.num_head, self.dim_head, self.dim_head)
            for _ in range(self.num_layers)
        )

        Wk_state_tuple = tuple(
            torch.zeros(1, bsz, self.num_head, self.dim_head, self.dim_head)
            for _ in range(self.num_layers)
        )

        wb_state_tuple = tuple(
            torch.zeros(1, bsz, self.num_head, self.dim_head, 4)
            for _ in range(self.num_layers)
        )

        fw_state_tuple = tuple(
            torch.zeros(1, bsz, self.num_head, self.dim_head, self.dim_head)
            for _ in range(self.num_layers)
        )

        return (Wy_state_tuple, Wq_state_tuple, Wk_state_tuple, wb_state_tuple,
                fw_state_tuple)

    def forward(self, inputs, core_state):
        x = inputs["frame"]
        assert x.device is not 'cpu'
        T, B, *_ = x.shape
        x = torch.flatten(x, 0, 1)  # Merge time and batch.
        x = x.float() / 255.0

        res_input = None
        for i, fconv in enumerate(self.feat_convs):
            x = fconv(x)
            res_input = x
            x = self.resnet1[i](x)
            x += res_input
            res_input = x
            x = self.resnet2[i](x)
            x += res_input

        x = F.relu(x)
        x = x.view(T * B, -1)
        x = F.relu(self.fc(x))

        clipped_reward = torch.clamp(inputs["reward"], -1, 1).view(T * B, 1)
        core_input = torch.cat([x, clipped_reward], dim=-1)

        # recurrent component
        core_input = core_input.view(T, B, -1)
        core_output_list = []
        notdone = (~inputs["done"]).float()
        for input, nd in zip(core_input.unbind(), notdone.unbind()):
            # Reset core state to zero whenever an episode ended.
            # Make `done` broadcastable with (num_layers, B, D, D)
            # states:
            # nd = nd.view(1, -1, 1)
            # needs extra dim0 for compat w/ inference code
            nd = nd.view(1, -1, 1, 1, 1)
            # better reset?
            (Wy_s, Wq_s, Wk_s, wb_s, fw_s) = core_state
            Wy_s = nest.map(nd.mul, Wy_s)
            Wq_s = nest.map(nd.mul, Wq_s)
            Wk_s = nest.map(nd.mul, Wk_s)
            wb_s = nest.map(nd.mul, wb_s)
            fw_s = nest.map(nd.mul, fw_s)
            core_state = (Wy_s, Wq_s, Wk_s, wb_s, fw_s)

            output, core_state = self.core(input.unsqueeze(0), core_state)
            core_output_list.append(output)

        core_output = torch.flatten(torch.cat(core_output_list), 0, 1)

        policy_logits = self.policy(core_output)
        baseline = self.baseline(core_output)

        if self.training:
            action = torch.multinomial(F.softmax(policy_logits, dim=1),
                                       num_samples=1)
        else:
            # Don't sample when testing.
            action = torch.argmax(policy_logits, dim=1)

        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)

        return (action, policy_logits, baseline), core_state

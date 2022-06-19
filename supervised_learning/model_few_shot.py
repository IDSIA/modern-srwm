# Implement models for few shot image classification
# NB: the current implementation uses one-hot encoding for label feedback
# (it might make sense to replace it by a regular embedding layer)
import torch
import torch.nn as nn

from layer import FastFFlayer, TransformerFFlayers, SRWMlayer
from resnet_impl import resnet12_base


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    # return number of parameters
    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def reset_grad(self):
        # More efficient than optimizer.zero_grad() according to:
        # Szymon Migacz "PYTORCH PERFORMANCE TUNING GUIDE" at GTC-21.
        # - doesn't execute memset for every parameter
        # - memory is zeroed-out by the allocator in a more efficient way
        # - backward pass updates gradients with "=" operator (write) (unlike
        # zero_grad() which would result in "+=").
        # In PyT >= 1.7, one can do `model.zero_grad(set_to_none=True)`
        for p in self.parameters():
            p.grad = None

    def print_params(self):
        for p in self.named_parameters():
            print(p)


# Conv4 by Vynials et al:
# '''
# We used a simple yet powerful CNN as the embedding function – consisting of
# a stack of modules, each of which is a 3×3 convolution with 64 filters
# followed by batch normalization [10], a Relu non-linearity and 2×2
# max-pooling. We resized all the images to 28 × 28 so that, when we stack 4
# modules, the resulting feature map is 1 × 1 × 64, resulting in our embedding
# function f(x).
# '''
class ConvLSTMModel(BaseModel):
    def __init__(self, hidden_size, num_classes, num_layer=1, imagenet=False,
                 fc100=False, vision_dropout=0.0, bn_momentum=0.1):
        super(ConvLSTMModel, self).__init__()

        num_conv_blocks = 4
        if imagenet:  # mini-imagenet
            input_channels = 3
            out_num_channel = 32
            self.conv_feature_final_size = 32 * 5 * 5  # (B, 32, 5, 5)
        elif fc100:
            input_channels = 3
            out_num_channel = 32
            self.conv_feature_final_size = 32 * 2 * 2  # (B, 32, 2, 2)
        else:  # onmiglot
            input_channels = 1
            out_num_channel = 64
            self.conv_feature_final_size = 64  # final feat shape (B, 64, 1, 1)

        self.input_channels = input_channels
        self.num_classes = num_classes
        list_conv_layers = []

        for i in range(num_conv_blocks):
            conv_block = []
            conv_block.append(
                nn.Conv2d(
                    in_channels=input_channels,
                    out_channels=out_num_channel,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            conv_block.append(nn.BatchNorm2d(
                out_num_channel, momentum=bn_momentum))
            conv_block.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
            conv_block.append(nn.Dropout(vision_dropout))
            conv_block.append(nn.ReLU(inplace=True))
            list_conv_layers.append(nn.Sequential(*conv_block))
            input_channels = out_num_channel

        self.conv_layers = nn.ModuleList(list_conv_layers)

        self.rnn = nn.LSTM(self.conv_feature_final_size + num_classes,
                           hidden_size, num_layers=num_layer)

        self.out_layer = nn.Linear(hidden_size, num_classes)

    def forward(self, x, fb, state=None):
        # Assume input of shape (len, B, 1, 28, 28)

        slen, bsz, _, hs, ws = x.shape
        x = x.reshape(slen * bsz, self.input_channels, hs, ws)

        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        x = x.reshape(slen, bsz, self.conv_feature_final_size)

        # alternatively use token embedding
        emb = torch.nn.functional.one_hot(fb, num_classes=self.num_classes)
        out = torch.cat([x, emb], dim=-1)

        out, _ = self.rnn(out, state)
        out = self.out_layer(out)

        return out, None


class ConvDeltaModel(BaseModel):
    def __init__(self, hidden_size, num_classes, num_layers, num_head,
                 dim_head, dim_ff, dropout, vision_dropout=0.0,
                 imagenet=False, fc100=False, bn_momentum=0.1):
        super(ConvDeltaModel, self).__init__()

        num_conv_blocks = 4
        if imagenet:  # mini-imagenet
            input_channels = 3
            out_num_channel = 32
            self.conv_feature_final_size = 32 * 5 * 5  # (B, 32, 5, 5)
        elif fc100:
            input_channels = 3
            out_num_channel = 32
            self.conv_feature_final_size = 32 * 2 * 2  # (B, 32, 5, 5)
        else:  # onmiglot
            input_channels = 1
            out_num_channel = 64
            self.conv_feature_final_size = 64  # final feat shape (B, 64, 1, 1)

        self.input_channels = input_channels
        self.num_classes = num_classes
        list_conv_layers = []

        for _ in range(num_conv_blocks):
            conv_block = []
            conv_block.append(
                nn.Conv2d(
                    in_channels=input_channels,
                    out_channels=out_num_channel,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            conv_block.append(nn.BatchNorm2d(
                out_num_channel, momentum=bn_momentum))
            conv_block.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
            conv_block.append(nn.Dropout(vision_dropout))
            conv_block.append(nn.ReLU(inplace=True))
            list_conv_layers.append(nn.Sequential(*conv_block))
            input_channels = out_num_channel

        self.conv_layers = nn.ModuleList(list_conv_layers)

        self.input_proj = nn.Linear(
            self.conv_feature_final_size + num_classes, hidden_size)

        layers = []

        for _ in range(num_layers):  # each "layer" consists of two sub-layers
            layers.append(
                FastFFlayer(num_head, dim_head, hidden_size, dropout))
            layers.append(
                TransformerFFlayers(dim_ff, hidden_size, dropout))
        self.layers = nn.Sequential(*layers)

        self.out_layer = nn.Linear(hidden_size, num_classes)

    def forward(self, x, fb, state=None):
        # Assume input of shape (len, B, 1, 28, 28)

        slen, bsz, _, hs, ws = x.shape
        x = x.reshape(slen * bsz, self.input_channels, hs, ws)

        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        x = x.reshape(slen, bsz, self.conv_feature_final_size)
        emb = torch.nn.functional.one_hot(fb, num_classes=self.num_classes)
        out = torch.cat([x, emb], dim=-1)

        out = self.input_proj(out)
        out = self.layers(out)
        out = self.out_layer(out)

        return out, None


class ConvSRWMModel(BaseModel):
    def __init__(self, hidden_size, num_classes, num_layers, num_head,
                 dim_head, dim_ff, dropout, vision_dropout=0.0,
                 use_ln=True, use_input_softmax=False, beta_init=0.,
                 imagenet=False, fc100=False, bn_momentum=0.1):
        super(ConvSRWMModel, self).__init__()

        num_conv_blocks = 4
        if imagenet:  # mini-imagenet
            input_channels = 3
            out_num_channel = 32
            self.conv_feature_final_size = 32 * 5 * 5  # (B, 32, 5, 5)
        elif fc100:
            input_channels = 3
            out_num_channel = 32
            self.conv_feature_final_size = 32 * 2 * 2  # (B, 32, 5, 5)
        else:  # onmiglot
            input_channels = 1
            out_num_channel = 64
            self.conv_feature_final_size = 64  # final feat shape (B, 64, 1, 1)

        self.input_channels = input_channels
        self.num_classes = num_classes
        list_conv_layers = []

        for _ in range(num_conv_blocks):
            conv_block = []
            conv_block.append(
                nn.Conv2d(
                    in_channels=input_channels,
                    out_channels=out_num_channel,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            conv_block.append(nn.BatchNorm2d(
                out_num_channel, momentum=bn_momentum))
            conv_block.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
            conv_block.append(nn.Dropout(vision_dropout))
            conv_block.append(nn.ReLU(inplace=True))
            list_conv_layers.append(nn.Sequential(*conv_block))
            input_channels = out_num_channel

        self.conv_layers = nn.ModuleList(list_conv_layers)

        self.input_proj = nn.Linear(
            self.conv_feature_final_size + num_classes, hidden_size)

        layers = []

        for _ in range(num_layers):  # each "layer" consists of two sub-layers
            layers.append(
                SRWMlayer(num_head, dim_head, hidden_size, dropout, use_ln,
                          use_input_softmax, beta_init))
            layers.append(
                TransformerFFlayers(dim_ff, hidden_size, dropout))
        self.layers = nn.Sequential(*layers)

        self.out_layer = nn.Linear(hidden_size, num_classes)

    def forward(self, x, fb, state=None):
        # Assume input of shape (len, B, 1, 28, 28)

        slen, bsz, _, hs, ws = x.shape
        x = x.reshape(slen * bsz, self.input_channels, hs, ws)

        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        x = x.reshape(slen, bsz, self.conv_feature_final_size)
        emb = torch.nn.functional.one_hot(fb, num_classes=self.num_classes)
        out = torch.cat([x, emb], dim=-1)

        out = self.input_proj(out)
        out = self.layers(out)
        out = self.out_layer(out)

        return out, None


class Res12LSTMModel(BaseModel):
    def __init__(self, hidden_size, num_classes,
                 num_layers, dropout, vision_dropout=0.0, use_big=False,
                 input_dropout=0.0, dropout_type='base'):
        super(Res12LSTMModel, self).__init__()

        self.stem_resnet12 = resnet12_base(
            vision_dropout, use_big, dropout_type)
        self.input_channels = 3
        self.num_classes = num_classes
        if use_big:
            self.conv_feature_final_size = 512
        else:
            self.conv_feature_final_size = 256

        self.input_drop = nn.Dropout(input_dropout)
        self.rnn = nn.LSTM(self.conv_feature_final_size + num_classes,
                           hidden_size, num_layers=num_layers,
                           dropout=dropout)

        self.out_layer = nn.Linear(hidden_size, num_classes)

    def forward(self, x, fb, state=None):
        # Assume input of shape (len, B, 1, 28, 28)

        slen, bsz, _, hs, ws = x.shape
        x = x.reshape(slen * bsz, self.input_channels, hs, ws)
        x = self.input_drop(x)

        x = self.stem_resnet12(x)

        x = x.reshape(slen, bsz, self.conv_feature_final_size)
        emb = torch.nn.functional.one_hot(fb, num_classes=self.num_classes)
        out = torch.cat([x, emb], dim=-1)

        out, _ = self.rnn(out, state)
        out = self.out_layer(out)

        return out, None


class Res12DeltaModel(BaseModel):
    def __init__(self, hidden_size, num_classes,
                 num_layers, num_head, dim_head, dim_ff,
                 dropout, vision_dropout=0.0, use_big=False,
                 input_dropout=0.0, dropout_type='base'):
        super(Res12DeltaModel, self).__init__()

        self.stem_resnet12 = resnet12_base(
            vision_dropout, use_big, dropout_type)
        self.input_channels = 3
        self.num_classes = num_classes
        if use_big:
            self.conv_feature_final_size = 512
        else:
            self.conv_feature_final_size = 256

        self.input_drop = nn.Dropout(input_dropout)
        self.input_proj = nn.Linear(
            self.conv_feature_final_size + num_classes, hidden_size)

        layers = []

        for _ in range(num_layers):  # each "layer" consists of two sub-layers
            layers.append(
                FastFFlayer(num_head, dim_head, hidden_size, dropout))
            layers.append(
                TransformerFFlayers(dim_ff, hidden_size, dropout))
        self.layers = nn.Sequential(*layers)

        self.out_layer = nn.Linear(hidden_size, num_classes)

    def forward(self, x, fb, state=None):
        # Assume input of shape (len, B, 1, 28, 28)

        slen, bsz, _, hs, ws = x.shape
        x = x.reshape(slen * bsz, self.input_channels, hs, ws)
        x = self.input_drop(x)

        x = self.stem_resnet12(x)

        x = x.reshape(slen, bsz, self.conv_feature_final_size)
        emb = torch.nn.functional.one_hot(fb, num_classes=self.num_classes)
        out = torch.cat([x, emb], dim=-1)

        out = self.input_proj(out)
        out = self.layers(out)
        out = self.out_layer(out)

        return out, None


class Res12SRWMModel(BaseModel):
    def __init__(self, hidden_size, num_classes, num_layers, num_head,
                 dim_head, dim_ff, dropout, vision_dropout=0.0,
                 use_big=False, use_ln=True, use_input_softmax=False,
                 input_dropout=0.0, dropout_type='base', beta_init=0.):
        super(Res12SRWMModel, self).__init__()

        self.stem_resnet12 = resnet12_base(
            vision_dropout, use_big, dropout_type)
        self.input_channels = 3
        self.num_classes = num_classes
        if use_big:
            self.conv_feature_final_size = 512
        else:
            self.conv_feature_final_size = 256

        self.input_drop = nn.Dropout(input_dropout)
        self.input_proj = nn.Linear(
            self.conv_feature_final_size + num_classes, hidden_size)

        layers = []

        for _ in range(num_layers):  # each "layer" consists of two sub-layers
            layers.append(
                SRWMlayer(num_head, dim_head, hidden_size, dropout, use_ln,
                          use_input_softmax, beta_init))
            layers.append(
                TransformerFFlayers(dim_ff, hidden_size, dropout))
        self.layers = nn.Sequential(*layers)

        self.out_layer = nn.Linear(hidden_size, num_classes)

    def forward(self, x, fb, state=None):
        # Assume input of shape (len, B, 1, 28, 28)

        slen, bsz, _, hs, ws = x.shape
        x = x.reshape(slen * bsz, self.input_channels, hs, ws)
        x = self.input_drop(x)

        x = self.stem_resnet12(x)

        x = x.reshape(slen, bsz, self.conv_feature_final_size)
        emb = torch.nn.functional.one_hot(fb, num_classes=self.num_classes)
        out = torch.cat([x, emb], dim=-1)

        out = self.input_proj(out)
        out = self.layers(out)
        out = self.out_layer(out)

        return out, None

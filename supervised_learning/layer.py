# Contain basic layers
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fast_weight import fast_weight_delta
from self_ref_v0 import self_ref_v0


@torch.jit.script
def elu_p1(x):
    return F.elu(x, 1., False) + 1.


@torch.jit.script
def sum_norm(x):
    return x / x.sum(-1, keepdim=True)


# A block of residual feed-forward layers in Transformer
class TransformerFFlayers(nn.Module):
    def __init__(self, ff_dim, res_dim, dropout, use_layernorm=True):
        super(TransformerFFlayers, self).__init__()

        self.res_dim = res_dim
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.use_layernorm = use_layernorm

        self.ff_layers = nn.Sequential(
            nn.Linear(res_dim, ff_dim), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, res_dim),
            nn.Dropout(dropout),
        )

        if use_layernorm:
            self.layer_norm = nn.LayerNorm(res_dim)

    def forward(self, x):
        out = self.layer_norm(x) if self.use_layernorm else x
        out = self.ff_layers(out) + x
        return out


# Fast weight layer with feed-forward fast net
class FastFFlayer(nn.Module):
    def __init__(self, num_head, dim_head, in_dim, dropout):
        super(FastFFlayer, self).__init__()

        self.num_head = num_head
        self.dim_head = dim_head
        self.in_dim = in_dim

        self.fw_layer = fast_weight_delta

        self.slow_net = nn.Linear(
            in_dim, num_head * (3 * dim_head + 1), bias=False)

        self.layer_norm = nn.LayerNorm(in_dim)
        self.out_linear = nn.Linear(num_head * dim_head, in_dim, bias=False)

        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (len, B, n_head * d_head)
        slen, bsz, _ = x.size()

        out = self.layer_norm(x)

        qkvb = self.slow_net(out)
        qkvb = qkvb.view(slen, bsz, self.num_head, 3 * self.dim_head + 1)
        head_q, head_k, head_v, head_beta = torch.split(
            qkvb, (self.dim_head,) * 3 + (1,), -1)
        head_beta = torch.sigmoid(head_beta)

        # reshape to (B, heads, len, dim)
        head_q = head_q.permute(1, 2, 0, 3)
        head_k = head_k.permute(1, 2, 0, 3)
        head_v = head_v.permute(1, 2, 0, 3)
        head_beta = head_beta.permute(1, 2, 0, 3)

        head_q = elu_p1(head_q)
        head_k = elu_p1(head_k)

        # normalize k and q, crucial for stable training.
        head_k = sum_norm(head_k)
        head_q = sum_norm(head_q)

        fast_weights = torch.zeros(
            bsz, self.num_head, self.dim_head, self.dim_head,
            device=head_k.device)

        out = self.fw_layer(head_q, head_k, head_v, head_beta, fast_weights)

        out = out.transpose(1, 2)
        out = out.reshape(bsz, slen, self.num_head * self.dim_head)
        out = out.transpose(0, 1)
        # expect [qlen, B, n_head * d_head]

        # linear projection
        out = self.out_linear(out)
        out = self.drop(out)
        out = x + out

        return out


# self referential weight matrix layer
class SRWMlayer(nn.Module):
    def __init__(self, num_head, dim_head, in_dim, dropout, use_ln=True,
                 use_input_softmax=False, beta_init=-1.0):
        super(SRWMlayer, self).__init__()

        self.num_head = num_head
        self.dim_head = dim_head
        self.in_dim = in_dim
        self.use_ln = use_ln
        self.use_input_softmax = use_input_softmax

        self.sr_layer = self_ref_v0
        n_head = num_head
        d_head = dim_head

        self.W_y = nn.Parameter(torch.Tensor(1, n_head, d_head, d_head),
                                requires_grad=True)
        self.W_q = nn.Parameter(torch.Tensor(1, n_head, d_head, d_head),
                                requires_grad=True)
        self.W_k = nn.Parameter(torch.Tensor(1, n_head, d_head, d_head),
                                requires_grad=True)
        self.w_b = nn.Parameter(torch.Tensor(1, n_head, d_head, 4),
                                requires_grad=True)
        if use_ln:
            self.layer_norm = nn.LayerNorm(in_dim)
        self.out_linear = nn.Linear(num_head * dim_head, in_dim, bias=False)

        self.drop = nn.Dropout(dropout)

        self.reset_parameters(beta_init)

    def reset_parameters(self, beta_init):
        std = 1.0 / math.sqrt(self.dim_head)
        nn.init.normal_(self.W_y, mean=0., std=std)
        nn.init.normal_(self.W_q, mean=0., std=std)
        nn.init.normal_(self.W_k, mean=0., std=std)
        # tried -1 for beta but 0 seems to be better
        # nn.init.normal_(self.w_b, mean=-5., std=std)
        nn.init.normal_(self.w_b, mean=beta_init, std=std)

    def forward(self, h, state=None, get_state=False):
        # x shape: (len, B, n_head * d_head)
        slen, bsz, _ = h.size()

        # out = self.layer_norm(x)
        x = h.reshape(slen, bsz, self.num_head, self.dim_head)
        if self.use_input_softmax:
            x = F.softmax(x, dim=-1)
        # reshape to (B, heads, len, dim)
        x = x.permute(1, 2, 0, 3)

        if state is not None:  # state stores the shift from the base weights.
            W_y_bc, W_q_bc, W_k_bc, w_b_bc = state
            W_y_bc = W_y_bc + self.W_y.repeat(bsz, 1, 1, 1)
            W_q_bc = W_q_bc + self.W_q.repeat(bsz, 1, 1, 1)
            W_k_bc = W_k_bc + self.W_k.repeat(bsz, 1, 1, 1)
            w_b_bc = w_b_bc + self.w_b.repeat(bsz, 1, 1, 1)
        else:
            W_y_bc = self.W_y.repeat(bsz, 1, 1, 1)
            W_q_bc = self.W_q.repeat(bsz, 1, 1, 1)
            W_k_bc = self.W_k.repeat(bsz, 1, 1, 1)
            w_b_bc = self.w_b.repeat(bsz, 1, 1, 1)

        out = self.sr_layer(x, W_y_bc, W_q_bc, W_k_bc, w_b_bc)

        out = out.transpose(1, 2)
        out = out.reshape(bsz, slen, self.num_head * self.dim_head)
        out = out.transpose(0, 1)
        # expect [qlen, B, n_head * d_head]

        # linear projection
        out = self.out_linear(out)
        out = self.drop(out)
        if self.use_ln:
            out = self.layer_norm(h) + out
        else:
            out = h + out
        # out = self.layer_norm(h) + out

        # compute the new shift (not very efficient; get it better from kernel)
        # if state is not None and get_state:
        if get_state:
            W_y_bc = W_y_bc.detach() - self.W_y.repeat(bsz, 1, 1, 1)
            W_q_bc = W_q_bc.detach() - self.W_q.repeat(bsz, 1, 1, 1)
            W_k_bc = W_k_bc.detach() - self.W_k.repeat(bsz, 1, 1, 1)
            w_b_bc = w_b_bc.detach() - self.w_b.repeat(bsz, 1, 1, 1)

            state = (
                W_y_bc.detach(), W_q_bc.detach(), W_k_bc.detach(),
                w_b_bc.detach())

            return out, state

        return out

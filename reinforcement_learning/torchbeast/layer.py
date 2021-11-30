import math

import torch
from torch import nn
from torch.nn import functional as F

from torchbeast.fast_weight import fast_weight_delta
from torchbeast.fast_transformers import fast_weight_sum
from torchbeast.rec_update_fwm_tanh import rec_update_fwm_tanh
from torchbeast.fast_weight_rnn_v2 import fast_rnn_v2
from torchbeast.self_ref_v0 import self_ref_v0
from torchbeast.self_ref_v1 import self_ref_v1


@torch.jit.script
def elu_p1(x):
    return F.elu(x, 1., False) + 1.


@torch.jit.script
def sum_norm(x):
    return x / x.sum(-1, keepdim=True)


@torch.jit.script
def sum_norm_eps(x):
    return x / (x.sum(-1, keepdim=True) + 1e-5)


@torch.jit.script
def elu_p1_sum_norm_eps(x):
    y = F.elu(x, 1., False) + 1.
    return y / (y.sum(-1, keepdim=True) + 1e-5)


# A block of residual feed-forward layers in Transformer
class TransformerFFlayers(nn.Module):
    def __init__(self, ff_dim, res_dim, dropout, use_layernorm=True):
        super(TransformerFFlayers, self).__init__()

        self.res_dim = res_dim
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.use_layernorm = use_layernorm

        self.ff_layers = nn.Sequential(
            nn.Linear(res_dim, ff_dim), nn.ReLU(inplace=False),
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
# linear tranformer
class AdditiveFastFFlayer(nn.Module):
    def __init__(self, num_head, dim_head, in_dim, dropout):
        super(AdditiveFastFFlayer, self).__init__()

        self.num_head = num_head
        self.dim_head = dim_head
        self.in_dim = in_dim

        self.fw_layer = fast_weight_sum

        self.slow_net = nn.Linear(
            in_dim, num_head * (3 * dim_head), bias=False)

        self.layer_norm = nn.LayerNorm(in_dim)
        self.out_linear = nn.Linear(num_head * dim_head, in_dim, bias=False)

        self.drop = nn.Dropout(dropout)

    def forward(self, x, state=None):
        # x shape: (len, B, n_head * d_head)
        slen, bsz, _ = x.size()

        out = self.layer_norm(x)

        qkv = self.slow_net(out)
        qkv = qkv.view(slen, bsz, self.num_head, 3 * self.dim_head)
        head_q, head_k, head_v = torch.split(qkv, (self.dim_head,) * 3, -1)

        # reshape to (B, heads, len, dim)
        head_q = head_q.permute(1, 2, 0, 3)
        head_k = head_k.permute(1, 2, 0, 3)
        head_v = head_v.permute(1, 2, 0, 3)

        head_q = elu_p1_sum_norm_eps(head_q)
        head_k = elu_p1_sum_norm_eps(head_k)

        if state is not None:
            fast_weights = state
        else:
            assert False
            fast_weights = torch.zeros(
                bsz, self.num_head, self.dim_head, self.dim_head,
                device=head_k.device)
        assert torch.isnan(
            fast_weights).sum().item() == 0, f"Before NaN: fast weights"

        out = self.fw_layer(head_q, head_k, head_v, fast_weights)

        assert torch.isnan(
            fast_weights).sum().item() == 0, f"NaN: fast weights"
        out = out.transpose(1, 2)
        out = out.reshape(bsz, slen, self.num_head * self.dim_head)
        out = out.transpose(0, 1)
        # expect [qlen, B, n_head * d_head]

        # linear projection
        out = self.out_linear(out)
        out = self.drop(out)
        out = x + out

        return out, fast_weights.clone()


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

    def forward(self, x, state=None):
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

        head_q = elu_p1_sum_norm_eps(head_q)
        head_k = elu_p1_sum_norm_eps(head_k)

        if state is not None:
            fast_weights = state
        else:
            assert False
            fast_weights = torch.zeros(
                bsz, self.num_head, self.dim_head, self.dim_head,
                device=head_k.device)
        assert torch.isnan(
            fast_weights).sum().item() == 0, f"Before NaN: fast weights"
        out = self.fw_layer(head_q, head_k, head_v, head_beta, fast_weights)
        assert torch.isnan(
            fast_weights).sum().item() == 0, f"NaN: fast weights"
        out = out.transpose(1, 2)
        out = out.reshape(bsz, slen, self.num_head * self.dim_head)
        out = out.transpose(0, 1)
        # expect [qlen, B, n_head * d_head]

        # linear projection
        out = self.out_linear(out)
        out = self.drop(out)
        out = x + out

        return out, fast_weights.clone()


# Fast weight layer with feed-forward fast net
class FastFastFFlayer(nn.Module):
    def __init__(self, num_head, dim_head, in_dim, dropout):
        super(FastFastFFlayer, self).__init__()

        self.num_head = num_head
        self.dim_head = dim_head
        self.in_dim = in_dim

        self.fw_layer = fast_weight_memory
        self.very_fw_layer = fast_weight_memory

        self.cached_fast_weights = nn.Parameter(
            torch.zeros(1, self.num_head, self.dim_head,
                        3 * self.dim_head + 1),
            requires_grad=False)

        self.slow_net = nn.Linear(
            in_dim, num_head * (5 * dim_head + 2), bias=False)

        self.layer_norm = nn.LayerNorm(in_dim)
        self.out_linear = nn.Linear(num_head * dim_head, in_dim, bias=False)

        self.drop = nn.Dropout(dropout)

    def forward(self, x, state=None):
        # x shape: (len, B, n_head * d_head)
        slen, bsz, _ = x.size()

        out = self.layer_norm(x)

        qkvb = self.slow_net(out)
        qkvb = qkvb.view(slen, bsz, self.num_head, 5 * self.dim_head + 2)
        head_q, head_k, head_v, head_beta = torch.split(
            qkvb, (self.dim_head, self.dim_head, 3 * self.dim_head + 1, 1), -1)
        head_beta = torch.sigmoid(head_beta)

        # reshape to (B, heads, len, dim)
        head_q = head_q.permute(1, 2, 0, 3)
        head_k = head_k.permute(1, 2, 0, 3)
        head_v = head_v.permute(1, 2, 0, 3)
        head_beta = head_beta.permute(1, 2, 0, 3)

        head_q = elu_p1_sum_norm_eps(head_q)
        head_k = elu_p1_sum_norm_eps(head_k)

        if state is not None:
            fast_weights, very_fast_weights = state
        else:
            assert False
            # For the delta-delta, this one is expected to be carried
            # over across episodes.
            fast_weights = torch.zeros(
                bsz, self.num_head, self.dim_head, 3 * self.dim_head + 1,
                device=head_k.device)
            # For delta
            very_fast_weights = torch.zeros(
                bsz, self.num_head, self.dim_head, self.dim_head,
                device=head_k.device)

        assert torch.isnan(
            fast_weights).sum().item() == 0, "Before NaN: fast weights"

        # forward fast weight
        out = self.fw_layer(head_q, head_k, head_v, head_beta, fast_weights)

        assert torch.isnan(
            fast_weights).sum().item() == 0, "NaN: fast weights"

        fast_head_q, fast_head_k, fast_head_v, fast_beta = torch.split(
            out, (self.dim_head,) * 3 + (1,), -1)

        fast_head_q = elu_p1_sum_norm_eps(fast_head_q)
        fast_head_k = elu_p1_sum_norm_eps(fast_head_k)
        fast_beta = torch.sigmoid(fast_beta)

        # forward very fast weight
        out = self.very_fw_layer(fast_head_q, fast_head_k, fast_head_v,
                                 fast_beta, very_fast_weights)

        out = out.transpose(1, 2)
        out = out.reshape(bsz, slen, self.num_head * self.dim_head)
        out = out.transpose(0, 1)
        # expect [qlen, B, n_head * d_head]

        # linear projection
        out = self.out_linear(out)
        out = self.drop(out)
        out = x + out

        return out, (fast_weights.clone(), very_fast_weights.clone())


# self referential weight matrix
class PseudoSRlayer(nn.Module):
    def __init__(self, num_head, dim_head, in_dim, dropout):
        super(PseudoSRlayer, self).__init__()

        self.num_head = num_head
        self.dim_head = dim_head
        self.in_dim = in_dim

        n_head = num_head
        d_head = dim_head

        self.W_y = nn.Parameter(torch.Tensor(n_head, d_head, d_head),
                                requires_grad=True)

        self.layer_norm = nn.LayerNorm(in_dim)
        self.out_linear = nn.Linear(num_head * dim_head, in_dim, bias=False)

        self.drop = nn.Dropout(dropout)

        self.reset_rec_parameters()

    def reset_rec_parameters(self):
        std = 1.0 / math.sqrt(self.dim_head)
        nn.init.normal_(self.W_y, mean=0., std=std)

    def forward(self, h, state=None):
        # x shape: (len, B, n_head * d_head)
        slen, bsz, _ = h.size()

        x = h.reshape(slen, bsz, self.num_head, self.dim_head)

        x = x.reshape(slen * bsz, self.num_head, self.dim_head)

        x = x.permute(1, 0, 2)  # (H, len*B, dim)

        out = torch.bmm(x, self.W_y)  # (H, len*B, dim)
        out = out.permute(1, 0, 2)  # (len*B, H, dim)
        out = out.reshape(slen, bsz, self.num_head, self.dim_head)
        out = out.reshape(slen, bsz, self.num_head * self.dim_head)

        # expect [qlen, B, n_head * d_head]

        # linear projection
        out = self.out_linear(out)
        out = self.drop(out)
        out = h + out

        # compute the new shift (not very efficient; get it better from kernel)

        # there is no need for extra state here...
        return out, state


# self referential weight matrix
class SRlayer(nn.Module):
    def __init__(self, num_head, dim_head, in_dim, dropout):
        super(SRlayer, self).__init__()

        self.num_head = num_head
        self.dim_head = dim_head
        self.in_dim = in_dim

        self.fw_layer = self_ref_v0
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

        self.layer_norm = nn.LayerNorm(in_dim)
        self.out_linear = nn.Linear(num_head * dim_head, in_dim, bias=False)

        self.drop = nn.Dropout(dropout)

        self.reset_rec_parameters()

    def reset_rec_parameters(self):
        std = 1.0 / math.sqrt(self.dim_head)
        nn.init.normal_(self.W_y, mean=0., std=std)
        nn.init.normal_(self.W_q, mean=0., std=std)
        nn.init.normal_(self.W_k, mean=0., std=std)
        # tried -1 for beta but 0 seems to be better
        # nn.init.normal_(self.w_b, mean=-5., std=std)
        nn.init.normal_(self.w_b, mean=0., std=std)

    def forward(self, h, state=None):
        # x shape: (len, B, n_head * d_head)
        slen, bsz, _ = h.size()

        # out = self.layer_norm(x)
        x = h.reshape(slen, bsz, self.num_head, self.dim_head)
        x = F.softmax(x, dim=-1)
        # reshape to (B, heads, len, dim)
        x = x.permute(1, 2, 0, 3)

        if state is not None:  # state store the shift from the current base weights.
            W_y_bc, W_q_bc, W_k_bc, w_b_bc = state
        else:
            assert False

        W_y_bc = W_y_bc + self.W_y.repeat(bsz, 1, 1, 1)
        W_q_bc = W_q_bc + self.W_q.repeat(bsz, 1, 1, 1)
        W_k_bc = W_k_bc + self.W_k.repeat(bsz, 1, 1, 1)
        w_b_bc = w_b_bc + self.w_b.repeat(bsz, 1, 1, 1)

        out = self.fw_layer(x, W_y_bc, W_q_bc, W_k_bc, w_b_bc)

        out = out.transpose(1, 2)
        out = out.reshape(bsz, slen, self.num_head * self.dim_head)
        out = out.transpose(0, 1)
        # expect [qlen, B, n_head * d_head]

        # linear projection
        out = self.out_linear(out)
        out = self.drop(out)
        out = h + out
        # out = self.layer_norm(h) + out

        # compute the new shift (not very efficient; get it better from kernel)

        W_y_bc = W_y_bc.detach() - self.W_y.repeat(bsz, 1, 1, 1)
        W_q_bc = W_q_bc.detach() - self.W_q.repeat(bsz, 1, 1, 1)
        W_k_bc = W_k_bc.detach() - self.W_k.repeat(bsz, 1, 1, 1)
        w_b_bc = w_b_bc.detach() - self.w_b.repeat(bsz, 1, 1, 1)

        state = (
            W_y_bc.detach(), W_q_bc.detach(), W_k_bc.detach(), w_b_bc.detach())

        # there is no need for extra state here...
        return out, state


# self referential weight matrix, without carrying weights over segment 
class NoCarryOverSRlayer(nn.Module):
    def __init__(self, num_head, dim_head, in_dim, dropout):
        super(NoCarryOverSRlayer, self).__init__()

        self.num_head = num_head
        self.dim_head = dim_head
        self.in_dim = in_dim

        self.fw_layer = self_ref_v0
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

        self.layer_norm = nn.LayerNorm(in_dim)
        self.out_linear = nn.Linear(num_head * dim_head, in_dim, bias=False)

        self.drop = nn.Dropout(dropout)

        self.reset_rec_parameters()

    def reset_rec_parameters(self):
        std = 1.0 / math.sqrt(self.dim_head)
        nn.init.normal_(self.W_y, mean=0., std=std)
        nn.init.normal_(self.W_q, mean=0., std=std)
        nn.init.normal_(self.W_k, mean=0., std=std)
        # tried -1 for beta but 0 seems to be better
        nn.init.normal_(self.w_b, mean=0., std=std)

    def forward(self, h, state=None):
        # x shape: (len, B, n_head * d_head)
        slen, bsz, _ = h.size()

        # out = self.layer_norm(x)
        x = h.reshape(slen, bsz, self.num_head, self.dim_head)
        x = F.softmax(x, dim=-1)
        # reshape to (B, heads, len, dim)
        x = x.permute(1, 2, 0, 3)

        W_y_bc = self.W_y.repeat(bsz, 1, 1, 1)
        W_q_bc = self.W_q.repeat(bsz, 1, 1, 1)
        W_k_bc = self.W_k.repeat(bsz, 1, 1, 1)
        w_b_bc = self.w_b.repeat(bsz, 1, 1, 1)

        out = self.fw_layer(x, W_y_bc, W_q_bc, W_k_bc, w_b_bc)
        out = out.transpose(1, 2)
        out = out.reshape(bsz, slen, self.num_head * self.dim_head)
        out = out.transpose(0, 1)
        # expect [qlen, B, n_head * d_head]

        # linear projection
        out = self.out_linear(out)
        out = self.drop(out)
        out = h + out

        # there is no need for extra state here...
        return out, state


# self modifying FWP layer
class SMFWPlayer(nn.Module):
    def __init__(self, num_head, dim_head, in_dim, dropout):
        super(SMFWPlayer, self).__init__()

        self.num_head = num_head
        self.dim_head = dim_head
        self.in_dim = in_dim

        self.fw_layer = self_ref_v1
        n_head = num_head
        d_head = dim_head

        y_d_head = 3 * dim_head + 1
        self.y_d_head = y_d_head

        self.W_y = nn.Parameter(torch.Tensor(1, n_head, d_head, y_d_head),
                                requires_grad=True)
        self.W_q = nn.Parameter(torch.Tensor(1, n_head, d_head, d_head),
                                requires_grad=True)
        self.W_k = nn.Parameter(torch.Tensor(1, n_head, d_head, d_head),
                                requires_grad=True)
        self.w_b = nn.Parameter(torch.Tensor(1, n_head, d_head, 4),
                                requires_grad=True)

        self.layer_norm = nn.LayerNorm(in_dim)
        self.out_linear = nn.Linear(num_head * dim_head, in_dim, bias=False)

        self.drop = nn.Dropout(dropout)

        self.reset_rec_parameters()

    def reset_rec_parameters(self):
        std = 1.0 / math.sqrt(self.dim_head)
        nn.init.normal_(self.W_y, mean=0., std=std)
        nn.init.normal_(self.W_q, mean=0., std=std)
        nn.init.normal_(self.W_k, mean=0., std=std)
        # tried -1 for beta but 0 seems to be better
        nn.init.normal_(self.w_b, mean=0., std=std)

    def forward(self, h, state=None):
        # x shape: (len, B, n_head * d_head)
        slen, bsz, _ = h.size()

        # out = self.layer_norm(x)
        x = h.reshape(slen, bsz, self.num_head, self.dim_head)
        x = F.softmax(x, dim=-1)
        # reshape to (B, heads, len, dim)
        x = x.permute(1, 2, 0, 3)

        if state is not None:  # state store the shift from the current base weights.
            W_y_bc, W_q_bc, W_k_bc, w_b_bc, fast_weights = state
        else:
            assert False
        assert torch.isnan(
            fast_weights).sum().item() == 0, "Before NaN: fast weights"

        W_y_bc = W_y_bc + self.W_y.repeat(bsz, 1, 1, 1)
        W_q_bc = W_q_bc + self.W_q.repeat(bsz, 1, 1, 1)
        W_k_bc = W_k_bc + self.W_k.repeat(bsz, 1, 1, 1)
        w_b_bc = w_b_bc + self.w_b.repeat(bsz, 1, 1, 1)

        fast_qkvb = self.fw_layer(x, W_y_bc, W_q_bc, W_k_bc, w_b_bc)

        fast_head_q, fast_head_k, fast_head_v, fast_beta = torch.split(
            fast_qkvb, (self.dim_head,) * 3 + (1,), -1)

        fast_head_q = F.softmax(fast_head_q, dim=-1)
        fast_head_k = F.softmax(fast_head_k, dim=-1)
        fast_beta = torch.sigmoid(fast_beta)

        out = fast_weight_memory(
            fast_head_q, fast_head_k, fast_head_v, fast_beta, fast_weights)

        out = out.transpose(1, 2)
        out = out.reshape(bsz, slen, self.num_head * self.dim_head)
        out = out.transpose(0, 1)
        # expect [qlen, B, n_head * d_head]

        # linear projection
        out = self.out_linear(out)
        out = self.drop(out)
        out = h + out
        # out = self.layer_norm(h) + out

        # compute the new shift (not very efficient; get it better from kernel)

        W_y_bc = W_y_bc.detach() - self.W_y.repeat(bsz, 1, 1, 1)
        W_q_bc = W_q_bc.detach() - self.W_q.repeat(bsz, 1, 1, 1)
        W_k_bc = W_k_bc.detach() - self.W_k.repeat(bsz, 1, 1, 1)
        w_b_bc = w_b_bc.detach() - self.w_b.repeat(bsz, 1, 1, 1)

        state = (
            W_y_bc.detach(), W_q_bc.detach(), W_k_bc.detach(), w_b_bc.detach(),
            fast_weights.detach())

        # there is no need for extra state here...
        return out, state


# Fast weight layer with RNN fast net
class FastRNNlayer(nn.Module):
    def __init__(self, num_head, dim_head, in_dim, dropout):
        super(FastRNNlayer, self).__init__()

        self.num_head = num_head
        self.dim_head = dim_head
        self.in_dim = in_dim
        self.fw_layer = fast_weight_delta
        self.rec_fw_layer = fast_rnn_v2

        self.slow_net = nn.Linear(
            in_dim, num_head * (5 * dim_head + 2), bias=False)

        self.layer_norm = nn.LayerNorm(in_dim)
        self.out_linear = nn.Linear(num_head * dim_head, in_dim, bias=False)

        self.drop = nn.Dropout(dropout)

    def forward(self, x, state=None):
        # x shape: (len, B, n_head * d_head)
        slen, bsz, _ = x.size()

        out = self.layer_norm(x)

        qkvb = self.slow_net(out)
        qkvb = qkvb.view(slen, bsz, self.num_head, 5 * self.dim_head + 2)
        (head_q, head_k, head_v, rec_head_k, rec_head_v, head_beta,
         rec_beta) = torch.split(qkvb, (self.dim_head,) * 5 + (1,) * 2, -1)

        head_beta = torch.sigmoid(head_beta)
        rec_beta = torch.sigmoid(rec_beta)

        # reshape to (B, heads, len, dim)
        head_q = head_q.permute(1, 2, 0, 3)
        head_k = head_k.permute(1, 2, 0, 3)
        head_v = head_v.permute(1, 2, 0, 3)
        head_beta = head_beta.permute(1, 2, 0, 3)

        rec_head_k = rec_head_k.permute(1, 2, 0, 3)
        rec_head_v = rec_head_v.permute(1, 2, 0, 3)
        rec_beta = rec_beta.permute(1, 2, 0, 3)

        head_q = F.softmax(head_q, dim=-1)
        head_k = F.softmax(head_k, dim=-1)
        # make recurrent key consistent with rec activation
        rec_head_k = F.softmax(rec_head_k, dim=-1)

        # # normalize k and q, crucial for stable training.
        # head_k = sum_norm(head_k)
        # head_q = sum_norm(head_q)

        if state is None:
            fast_weights = torch.zeros(
                bsz, self.num_head, self.dim_head, self.dim_head,
                device=head_k.device)

            rec_fast_weights = torch.zeros(
                bsz, self.num_head, self.dim_head, self.dim_head,
                device=head_k.device)

            state0 = torch.zeros(
                bsz, self.num_head, 1, self.dim_head, device=head_k.device)
        else:
            fast_weights, rec_fast_weights, state0 = state
        assert torch.isnan(
            fast_weights).sum().item() == 0, f"Before NaN: fast weights"
        z_out = self.fw_layer(
            head_q, head_k, head_v, head_beta, fast_weights)

        out = self.rec_fw_layer(
            z_out, rec_head_k, rec_head_v, rec_fast_weights, rec_beta, state0)

        state0_next = out[:, :, -1, :].clone()
        state0_next = state0_next.unsqueeze(2)

        out = out.transpose(1, 2)
        out = out.reshape(bsz, slen, self.num_head * self.dim_head)
        out = out.transpose(0, 1)
        # expect [qlen, B, n_head * d_head]

        # linear projection
        out = self.out_linear(out)
        out = self.drop(out)
        out = x + out

        return out, (
            fast_weights.clone(), rec_fast_weights.clone(), state0_next)


class RecUpdateTanhFastFFlayer(nn.Module):
    def __init__(self, num_head, dim_head, in_dim, dropout):
        super(RecUpdateTanhFastFFlayer, self).__init__()

        self.num_head = num_head
        self.dim_head = dim_head
        self.in_dim = in_dim

        self.fw_layer = rec_update_fwm_tanh

        self.slow_net = nn.Linear(
            in_dim, num_head * (3 * dim_head + 1), bias=False)

        self.R_q = nn.Parameter(torch.Tensor(1, num_head, dim_head, dim_head),
                                requires_grad=True)
        self.R_k = nn.Parameter(torch.Tensor(1, num_head, dim_head, dim_head),
                                requires_grad=True)
        self.R_v = nn.Parameter(torch.Tensor(1, num_head, dim_head, dim_head),
                                requires_grad=True)
        self.r_b = nn.Parameter(torch.Tensor(1, num_head, 1, dim_head),
                                requires_grad=True)

        self.layer_norm = nn.LayerNorm(in_dim)
        self.out_linear = nn.Linear(num_head * dim_head, in_dim, bias=False)

        self.drop = nn.Dropout(dropout)

        self.reset_rec_parameters()

    def reset_rec_parameters(self):
        std = 1.0 / math.sqrt(self.dim_head)
        nn.init.normal_(self.R_q, mean=0., std=std)
        nn.init.normal_(self.R_k, mean=0., std=std)
        nn.init.normal_(self.R_v, mean=0., std=std)
        nn.init.normal_(self.r_b, mean=0., std=std)

    def forward(self, x, state=None):
        # x shape: (len, B, n_head * d_head)
        slen, bsz, _ = x.size()

        out = self.layer_norm(x)

        qkvb = self.slow_net(out)
        qkvb = qkvb.view(slen, bsz, self.num_head, 3 * self.dim_head + 1)
        head_q, head_k, head_v, head_beta = torch.split(
            qkvb, (self.dim_head,) * 3 + (1,), -1)

        # reshape to (B, heads, len, dim)
        head_q = head_q.permute(1, 2, 0, 3)
        head_k = head_k.permute(1, 2, 0, 3)
        head_v = head_v.permute(1, 2, 0, 3)
        head_beta = head_beta.permute(1, 2, 0, 3)

        if state is None:
            fast_weights = torch.zeros(
                bsz, self.num_head, self.dim_head, self.dim_head,
                device=head_k.device)

            state0 = torch.zeros(
                bsz, self.num_head, 1, self.dim_head, device=head_k.device)
        else:
            fast_weights, state0 = state

        out = self.fw_layer(head_q, head_k, head_v, head_beta,
                            self.R_q, self.R_k, self.R_v, self.r_b,
                            fast_weights, state0)

        state0_next = out[:, :, -1, :].clone()
        state0_next = state0_next.unsqueeze(2)

        out = out.transpose(1, 2)
        out = out.reshape(bsz, slen, self.num_head * self.dim_head)
        out = out.transpose(0, 1)
        # expect [qlen, B, n_head * d_head]

        # linear projection
        out = self.out_linear(out)
        out = self.drop(out)
        out = x + out

        return out, (fast_weights.clone(), state0_next)


# Linear Transformer with Fast weight memory update rule.
class LinearTransformerLayer(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, num_head, dim_head,
                 dim_ff, dropout):
        super(LinearTransformerLayer, self).__init__()
        assert num_head * dim_head == hidden_size

        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_head = num_head
        self.dim_head = dim_head
        self.dim_ff = dim_ff
        self.dropout = dropout

        self.input_proj = nn.Linear(input_dim, hidden_size)

        fwm_layers = []
        ff_layers = []

        for _ in range(num_layers):  # each "layer" consists of two sub-layers
            fwm_layers.append(
                AdditiveFastFFlayer(num_head, dim_head, hidden_size, dropout))
            ff_layers.append(
                TransformerFFlayers(dim_ff, hidden_size, dropout))

        self.fwm_layers = nn.ModuleList(fwm_layers)
        self.ff_layers = nn.ModuleList(ff_layers)

    def forward(self, x, core_state):
        # core_state is a tuple with self.num_layers elements
        state_list = []

        out = self.input_proj(x)  # shape (len, B, dim)
        for i in range(self.num_layers):
            out, out_state = self.fwm_layers[i](out, state=core_state[i].squeeze(0))
            state_list.append(out_state.unsqueeze(0).clone())
            out = self.ff_layers[i](out)

        state_tuple = tuple(state_list)
        return out, state_tuple


# DeeperNet, Transformer block without self-attention
class DeeperNetLayer(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, dim_ff, dropout):
        super(DeeperNetLayer, self).__init__()

        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dim_ff = dim_ff
        self.dropout = dropout

        self.input_proj = nn.Linear(input_dim, hidden_size)

        ff_layers = []

        for _ in range(num_layers):  # each "layer" consists of two sub-layers
            ff_layers.append(
                TransformerFFlayers(dim_ff, hidden_size, dropout))

        self.ff_layers = nn.ModuleList(ff_layers)

    def forward(self, x):
        # core_state is a tuple with self.num_layers elements
        out = self.input_proj(x)  # shape (len, B, dim)
        for i in range(self.num_layers):
            out = self.ff_layers[i](out)
        return out


# Linear Transformer with Fast weight memory update rule.
class DeltaNetLayer(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, num_head, dim_head,
                 dim_ff, dropout):
        super(DeltaNetLayer, self).__init__()
        assert num_head * dim_head == hidden_size

        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_head = num_head
        self.dim_head = dim_head
        self.dim_ff = dim_ff
        self.dropout = dropout

        self.input_proj = nn.Linear(input_dim, hidden_size)

        fwm_layers = []
        ff_layers = []

        for _ in range(num_layers):  # each "layer" consists of two sub-layers
            fwm_layers.append(
                FastFFlayer(num_head, dim_head, hidden_size, dropout))
            ff_layers.append(
                TransformerFFlayers(dim_ff, hidden_size, dropout))

        self.fwm_layers = nn.ModuleList(fwm_layers)
        self.ff_layers = nn.ModuleList(ff_layers)

    def forward(self, x, core_state):
        # core_state is a tuple with self.num_layers elements
        state_list = []

        out = self.input_proj(x)  # shape (len, B, dim)

        for i in range(self.num_layers):
            out, out_state = self.fwm_layers[i](out, state=core_state[i].squeeze(0))
            state_list.append(out_state.unsqueeze(0).clone())
            out = self.ff_layers[i](out)

        state_tuple = tuple(state_list)
        return out, state_tuple


class FastFFRecUpdateTanhLayer(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, num_head, dim_head,
                 dim_ff, dropout):
        super(FastFFRecUpdateTanhLayer, self).__init__()
        assert num_head * dim_head == hidden_size

        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_head = num_head
        self.dim_head = dim_head
        self.dim_ff = dim_ff
        self.dropout = dropout

        self.input_proj = nn.Linear(input_dim, hidden_size)

        fwm_layers = []
        ff_layers = []

        for _ in range(num_layers):  # each "layer" consists of two sub-layers
            fwm_layers.append(
                RecUpdateTanhFastFFlayer(
                    num_head, dim_head, hidden_size, dropout))
            ff_layers.append(
                TransformerFFlayers(dim_ff, hidden_size, dropout))

        self.fwm_layers = nn.ModuleList(fwm_layers)
        self.ff_layers = nn.ModuleList(ff_layers)

    def forward(self, x, core_state):
        fw_states, rnn_states = core_state
        # core_state is a tuple with self.num_layers elements
        fw_state_list = []
        rnn_state_list = []

        out = self.input_proj(x)  # shape (len, B, dim)

        for i in range(self.num_layers):
            out, out_state = self.fwm_layers[i](
                out, state=(fw_states[i].squeeze(0), rnn_states[i].squeeze(0)))
            fw_state_list.append(out_state[0].unsqueeze(0).clone())
            rnn_state_list.append(out_state[1].unsqueeze(0).clone())
            out = self.ff_layers[i](out)

        fw_state_tuple = tuple(fw_state_list)
        rnn_state_tuple = tuple(rnn_state_list)
        state_tuple = (fw_state_tuple, rnn_state_tuple)
        return out, state_tuple


class FastRNNModelLayer(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, num_head, dim_head,
                 dim_ff, dropout):
        super(FastRNNModelLayer, self).__init__()
        assert num_head * dim_head == hidden_size

        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_head = num_head
        self.dim_head = dim_head
        self.dim_ff = dim_ff
        self.dropout = dropout

        self.input_proj = nn.Linear(input_dim, hidden_size)

        fwm_layers = []
        ff_layers = []

        for _ in range(num_layers):  # each "layer" consists of two sub-layers
            fwm_layers.append(
                FastRNNlayer(num_head, dim_head, hidden_size, dropout))
            ff_layers.append(
                TransformerFFlayers(dim_ff, hidden_size, dropout))

        self.fwm_layers = nn.ModuleList(fwm_layers)
        self.ff_layers = nn.ModuleList(ff_layers)

    def forward(self, x, core_state):
        fw_states, rec_fw_states, rnn_states = core_state
        # core_state is a tuple with self.num_layers elements
        fw_state_list = []
        rec_fw_state_list = []
        rnn_state_list = []

        out = self.input_proj(x)  # shape (len, B, dim)

        for i in range(self.num_layers):
            out, out_state = self.fwm_layers[i](
                out,
                state=(fw_states[i].squeeze(0), rec_fw_states[i].squeeze(0),
                       rnn_states[i].squeeze(0)))
            fw_state_list.append(out_state[0].unsqueeze(0).clone())
            rec_fw_state_list.append(out_state[1].unsqueeze(0).clone())
            rnn_state_list.append(out_state[2].unsqueeze(0).clone())
            out = self.ff_layers[i](out)

        fw_state_tuple = tuple(fw_state_list)
        rec_fw_state_tuple = tuple(rec_fw_state_list)
        rnn_state_tuple = tuple(rnn_state_list)
        state_tuple = (fw_state_tuple, rec_fw_state_tuple, rnn_state_tuple)
        return out, state_tuple


# delta delta.
class DeltaDeltaNetLayer(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, num_head, dim_head,
                 dim_ff, dropout):
        super(DeltaDeltaNetLayer, self).__init__()
        assert num_head * dim_head == hidden_size

        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_head = num_head
        self.dim_head = dim_head
        self.dim_ff = dim_ff
        self.dropout = dropout

        self.input_proj = nn.Linear(input_dim, hidden_size)

        fwm_layers = []
        ff_layers = []

        for _ in range(num_layers):  # each "layer" consists of two sub-layers
            fwm_layers.append(
                FastFastFFlayer(num_head, dim_head, hidden_size, dropout))
            ff_layers.append(
                TransformerFFlayers(dim_ff, hidden_size, dropout))

        self.fwm_layers = nn.ModuleList(fwm_layers)
        self.ff_layers = nn.ModuleList(ff_layers)

    def forward(self, x, core_state):
        fw_states, very_fw_states = core_state
        # core_state is a tuple with self.num_layers elements
        fw_state_list = []
        very_fw_state_list = []

        out = self.input_proj(x)  # shape (len, B, dim)

        for i in range(self.num_layers):
            out, out_state = self.fwm_layers[i](
                out,
                state=(fw_states[i].squeeze(0), very_fw_states[i].squeeze(0)))
            fw_state_list.append(out_state[0].unsqueeze(0).clone())
            very_fw_state_list.append(out_state[1].unsqueeze(0).clone())
            out = self.ff_layers[i](out)

        fw_state_tuple = tuple(fw_state_list)
        very_fw_state_tuple = tuple(very_fw_state_list)
        state_tuple = (fw_state_tuple, very_fw_state_tuple)

        return out, state_tuple


# Simple Self-Referential layer
class SRNetLayer(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, num_head, dim_head,
                 dim_ff, dropout):
        super(SRNetLayer, self).__init__()
        assert num_head * dim_head == hidden_size

        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_head = num_head
        self.dim_head = dim_head
        self.dim_ff = dim_ff
        self.dropout = dropout

        self.input_proj = nn.Linear(input_dim, hidden_size)

        fwm_layers = []
        ff_layers = []

        for _ in range(num_layers):  # each "layer" consists of two sub-layers
            fwm_layers.append(
                SRlayer(num_head, dim_head, hidden_size, dropout))
            ff_layers.append(
                TransformerFFlayers(dim_ff, hidden_size, dropout))

        self.fwm_layers = nn.ModuleList(fwm_layers)
        self.ff_layers = nn.ModuleList(ff_layers)

    def forward(self, x, core_state):
        Wy_states, Wq_states, Wk_states, wb_states = core_state
        # core_state is a tuple with self.num_layers elements
        Wy_state_list = []
        Wq_state_list = []
        Wk_state_list = []
        wb_state_list = []

        out = self.input_proj(x)  # shape (len, B, dim)

        for i in range(self.num_layers):
            out, out_state = self.fwm_layers[i](
                out,
                state=(Wy_states[i].squeeze(0), Wq_states[i].squeeze(0),
                       Wk_states[i].squeeze(0), wb_states[i].squeeze(0)))
            Wy_state_list.append(out_state[0].unsqueeze(0).clone())
            Wq_state_list.append(out_state[1].unsqueeze(0).clone())
            Wk_state_list.append(out_state[2].unsqueeze(0).clone())
            wb_state_list.append(out_state[3].unsqueeze(0).clone())
            out = self.ff_layers[i](out)

        Wy_state_tuple = tuple(Wy_state_list)
        Wq_state_tuple = tuple(Wq_state_list)
        Wk_state_tuple = tuple(Wk_state_list)
        wb_state_tuple = tuple(wb_state_list)

        state_tuple = (
            Wy_state_tuple, Wq_state_tuple, Wk_state_tuple, wb_state_tuple)

        return out, state_tuple


# Simple Self-Referential layer
class PseudoSRNetLayer(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, num_head, dim_head,
                 dim_ff, dropout):
        super(PseudoSRNetLayer, self).__init__()
        assert num_head * dim_head == hidden_size

        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_head = num_head
        self.dim_head = dim_head
        self.dim_ff = dim_ff
        self.dropout = dropout

        self.input_proj = nn.Linear(input_dim, hidden_size)

        fwm_layers = []
        ff_layers = []

        for _ in range(num_layers):  # each "layer" consists of two sub-layers
            fwm_layers.append(
                PseudoSRlayer(num_head, dim_head, hidden_size, dropout))
            ff_layers.append(
                TransformerFFlayers(dim_ff, hidden_size, dropout))

        self.fwm_layers = nn.ModuleList(fwm_layers)
        self.ff_layers = nn.ModuleList(ff_layers)

    def forward(self, x, core_state):
        out = self.input_proj(x)  # shape (len, B, dim)

        for i in range(self.num_layers):
            out, _ = self.fwm_layers[i](out)
            out = self.ff_layers[i](out)

        return out, core_state


# Simple Self-Referential layer, no carry over weights
class NoCarryOverSRNetLayer(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, num_head, dim_head,
                 dim_ff, dropout):
        super(NoCarryOverSRNetLayer, self).__init__()
        assert num_head * dim_head == hidden_size

        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_head = num_head
        self.dim_head = dim_head
        self.dim_ff = dim_ff
        self.dropout = dropout

        self.input_proj = nn.Linear(input_dim, hidden_size)

        fwm_layers = []
        ff_layers = []

        for _ in range(num_layers):  # each "layer" consists of two sub-layers
            fwm_layers.append(
                NoCarryOverSRlayer(num_head, dim_head, hidden_size, dropout))
            ff_layers.append(
                TransformerFFlayers(dim_ff, hidden_size, dropout))

        self.fwm_layers = nn.ModuleList(fwm_layers)
        self.ff_layers = nn.ModuleList(ff_layers)

    def forward(self, x, core_state):

        out = self.input_proj(x)  # shape (len, B, dim)

        for i in range(self.num_layers):
            out, _ = self.fwm_layers[i](out, state=core_state)
            out = self.ff_layers[i](out)

        return out, core_state


# Simple self-modifying FWP layer
class SMFWPNetLayer(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, num_head, dim_head,
                 dim_ff, dropout):
        super(SMFWPNetLayer, self).__init__()
        assert num_head * dim_head == hidden_size

        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_head = num_head
        self.dim_head = dim_head
        self.dim_ff = dim_ff
        self.dropout = dropout

        self.input_proj = nn.Linear(input_dim, hidden_size)

        fwm_layers = []
        ff_layers = []

        for _ in range(num_layers):  # each "layer" consists of two sub-layers
            fwm_layers.append(
                SMFWPlayer(num_head, dim_head, hidden_size, dropout))
            ff_layers.append(
                TransformerFFlayers(dim_ff, hidden_size, dropout))

        self.fwm_layers = nn.ModuleList(fwm_layers)
        self.ff_layers = nn.ModuleList(ff_layers)

    def forward(self, x, core_state):
        Wy_states, Wq_states, Wk_states, wb_states, fw_states = core_state
        # core_state is a tuple with self.num_layers elements
        Wy_state_list = []
        Wq_state_list = []
        Wk_state_list = []
        wb_state_list = []
        fw_state_list = []

        out = self.input_proj(x)  # shape (len, B, dim)

        for i in range(self.num_layers):
            out, out_state = self.fwm_layers[i](
                out,
                state=(Wy_states[i].squeeze(0), Wq_states[i].squeeze(0),
                       Wk_states[i].squeeze(0), wb_states[i].squeeze(0),
                       fw_states[i].squeeze(0))
            )
            Wy_state_list.append(out_state[0].unsqueeze(0).clone())
            Wq_state_list.append(out_state[1].unsqueeze(0).clone())
            Wk_state_list.append(out_state[2].unsqueeze(0).clone())
            wb_state_list.append(out_state[3].unsqueeze(0).clone())
            fw_state_list.append(out_state[4].unsqueeze(0).clone())
            out = self.ff_layers[i](out)

        Wy_state_tuple = tuple(Wy_state_list)
        Wq_state_tuple = tuple(Wq_state_list)
        Wk_state_tuple = tuple(Wk_state_list)
        wb_state_tuple = tuple(wb_state_list)
        fw_state_tuple = tuple(fw_state_list)

        state_tuple = (
            Wy_state_tuple, Wq_state_tuple, Wk_state_tuple, wb_state_tuple,
            fw_state_tuple)

        return out, state_tuple

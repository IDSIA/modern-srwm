# Adaptation of the original code from
# https://github.com/idiap/fast-transformers/blob/master/fast_transformers/causal_product/__init__.py
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#
# Modifications Copyright (c) 2021 Kazuki Irie

import os
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load
# Just in time import
# https://pytorch.org/tutorials/advanced/cpp_extens

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'fast_rnn_cuda.cu')

mod_causal_dot_product_cuda = load(
    name="fast_rnn_v2_forward",
    sources=[filename], verbose=True)
mod_causal_dot_backward_cuda = load(
    name="fast_rnn_v2_backward",
    sources=[filename], verbose=True)


causal_dot_product_cuda = mod_causal_dot_product_cuda.fast_rnn_v2_forward
causal_dot_backward_cuda = mod_causal_dot_backward_cuda.fast_rnn_v2_backward


class FastRNNv2(torch.autograd.Function):
    """Fast RNN with the FWM update rule."""
    dot = {
        # "cpu": causal_dot_product_cpu,
        "cuda": causal_dot_product_cuda
    }
    dot_backward = {
        # "cpu": causal_dot_backward_cpu,
        "cuda": causal_dot_backward_cuda
    }

    @staticmethod
    def forward(ctx, Z, K, V, W, beta, h0):
        # Computations:
        #   fast weights with sum update rule: R_t = R_t-1 + v_t (x) k_t
        #   output: h_t = tanh(R_t * h_t-1 + z_t)
        # z_t is the output of a feed-forward fast weight layer.
        # h0 is the initial RNN state.
        # E = M.

        # Create the output tensor
        device = Z.device
        N, H, L, _ = Z.shape
        _, _, _, M = V.shape

        assert K.shape == (N, H, L, M)
        assert V.shape == (N, H, L, M)
        assert h0.shape == (N, H, 1, M)
        assert W.shape == (N, H, M, M)

        rnn_out = torch.zeros((N, H, L, M), device=device, dtype=Z.dtype)
        rnn_out_nmz = torch.zeros((N, H, L, M), device=device, dtype=Z.dtype)
        h_init = h0.detach().clone()
        # W = torch.zeros((N, H, E, M), device=device, dtype=Z.dtype)
        # h0 = torch.zeros((N, H, M), device=device, dtype=Z.dtype)
        V_old = torch.zeros((N, H, L, M), device=device, dtype=Z.dtype)

        # Actually perform the dot product
        FastRNNv2.dot[device.type](
            Z.data,
            K.data,
            V.data,
            beta.data,
            V_old,
            h0.data,
            W,
            rnn_out,
            rnn_out_nmz
        )

        ctx.save_for_backward(rnn_out, rnn_out_nmz,
                              Z, K, V, beta, V_old, W, h_init)

        return rnn_out

    @staticmethod
    def backward(ctx, grad_out):
        # Extract the saved tensors
        rnn_out, rnn_out_nmz, Z, K, V, beta, V_old, W, h0 = ctx.saved_tensors

        # Allocate memory for the gradients
        grad_Z = torch.zeros_like(Z)
        grad_K = torch.zeros_like(K)
        grad_V = torch.zeros_like(V)
        grad_beta = torch.zeros_like(beta)

        # Prepare delayed RNN outputs
        # shape of rnn_out: N, H, L, M
        # dim2 is the time dim.
        # shape of h0: N, H, 1, M
        # rnn_out_delayed = torch.cat([h0, rnn_out[:, :, :-1]], dim=2)

        # Compute the gradients
        FastRNNv2.dot_backward[Z.device.type](
            K.data,
            V.data,
            beta.data,
            V_old.data,
            grad_out,
            rnn_out,
            rnn_out_nmz,  # normalized and delayed
            W.data,
            grad_Z,
            grad_K,
            grad_V,
            grad_beta,
        )

        return grad_Z, grad_K, grad_V, None, grad_beta, None


# Alias the autograd functions to python style snake case naming
fast_rnn_v2 = FastRNNv2.apply


if __name__ == '__main__':
    import torch
    torch.manual_seed(111)
    # Tests pass if the relative difference compared with
    # the corresponding torch autograd computation
    # is smaller than a threshold.

    # Ideally should be tested with double...
    rel_threshold = 1e-3

    # from https://github.com/idiap/fast-transformers/blob/master/tests/causal_product/test_causal_product_gpu.py
    def max_relative_error(a, b, eps=1e-6):
        return float(torch.abs((b - a) / (torch.abs(b) + eps)).max().item())

    print('##########################')
    print('# Test forward pass')
    print('##########################')

    bsz, n_head, slen, d_head = 3, 5, 11, 64
    v_dim = d_head

    # (B, H, len, dim)
    k0 = torch.rand(bsz, n_head, slen, d_head, device='cuda')
    v0 = torch.rand(bsz, n_head, slen, v_dim, device='cuda')
    beta0 = torch.sigmoid(torch.rand(bsz, n_head, slen, 1, device='cuda'))
    h0 = torch.zeros(bsz, n_head, 1, v_dim, device='cuda')
    z0 = torch.rand(bsz, n_head, slen, v_dim, device='cuda')

    # k0 = k0 / k0.sum(dim=-1, keepdim=True)
    k0 = F.softmax(k0, dim=-1)

    k1 = torch.zeros(
        bsz, n_head, slen, d_head, requires_grad=True, device='cuda')
    v1 = torch.zeros(
        bsz, n_head, slen, v_dim, requires_grad=True, device='cuda')
    z1 = torch.zeros(
        bsz, n_head, slen, v_dim, requires_grad=True, device='cuda')
    beta1 = torch.zeros(
        bsz, n_head, slen, 1, requires_grad=True, device='cuda')

    # q1.data = q0.data
    k1.data = k0.data
    v1.data = v0.data
    beta1.data = beta0.data
    z1.data = z0.data

    W1 = torch.zeros(bsz, n_head, d_head, v_dim, device='cuda')
    # h0 = torch.zeros(n_head, d_head, v_dim, device='cuda')
    print("Forwarding custom kernel...")
    out1 = fast_rnn_v2(z1, k1, v1, W1, beta1, h0)
    print("done.")

    # compute using torch
    z2 = torch.zeros(
        bsz, n_head, slen, d_head, requires_grad=True, device='cuda')
    k2 = torch.zeros(
        bsz, n_head, slen, d_head, requires_grad=True, device='cuda')
    v2 = torch.zeros(
        bsz, n_head, slen, v_dim, requires_grad=True, device='cuda')
    beta2 = torch.zeros(
        bsz, n_head, slen, 1, requires_grad=True, device='cuda')

    z2.data = z0.data
    k2.data = k0.data
    v2.data = v0.data
    beta2.data = beta0.data

    # (len, B, H, dim)
    z_2 = z2.permute(2, 0, 1, 3)
    slen, bsz, n_head, d_head = z_2.shape
    z_2 = z_2.reshape(slen, bsz * n_head, d_head)

    k_2 = k2.permute(2, 0, 1, 3)
    k_2 = k_2.reshape(slen, bsz * n_head, d_head)

    v_2 = v2.permute(2, 0, 1, 3)
    v_2 = v_2.reshape(slen, bsz * n_head, v_dim)

    beta_2 = beta2.permute(2, 0, 1, 3)
    beta_2 = beta_2.reshape(slen, bsz * n_head, 1)

    W = torch.zeros(bsz * n_head, v_dim, d_head, device='cuda')
    h = torch.zeros(bsz * n_head, d_head, device='cuda')

    out_list = []
    print("Forwarding PyTorch code...")
    for pos in range(slen):
        v_old = torch.bmm(W, k_2[pos].unsqueeze(2)).squeeze()
        v_insert = beta_2[pos] * (v_2[pos] - v_old)
        W = W + torch.bmm(v_insert.unsqueeze(2), k_2[pos].unsqueeze(1))
        rec_part = torch.bmm(W, F.softmax(h, dim=-1).unsqueeze(2)).squeeze()
        # h = torch.sigmoid(rec_part + z_2[pos])
        h = rec_part + z_2[pos]
        out_list.append(h.clone())
    print("done.")

    out2 = torch.stack(out_list)
    out2 = out2.view(slen, bsz, n_head, v_dim)

    out1 = out1.permute(2, 0, 1, 3)

    for s in range(slen):
        for b in range(bsz):
            for h in range(n_head):
                print(f"forward: s={s} b={b} h={h}")
                print(f"out: {out1[s][b][h]}")
                print(f"ref: {out2[s][b][h]}")
                assert max_relative_error(
                    out1[s][b][h], out2[s][b][h]) < rel_threshold
                print("pass!")

    print('##########################')
    print('# Test Backward pass')
    print('##########################')

    # grad
    loss1 = out1.sum()
    z1.retain_grad()
    k1.retain_grad()
    v1.retain_grad()
    beta1.retain_grad()

    loss1.backward()

    loss2 = out2.sum()
    z2.retain_grad()
    k2.retain_grad()
    v2.retain_grad()
    beta2.retain_grad()

    loss2.backward()

    for s in reversed(range(slen)):
        for b in reversed(range(bsz)):
            for h in range(n_head):
                print(f"backward: s={s}, b={b}, h={h}")
                print(f"grad input out: {z1.grad[b][h][s]}")
                print(f"grad input ref: {z2.grad[b][h][s]}")
                assert max_relative_error(
                    z1.grad[b][h][s], z2.grad[b][h][s]) < rel_threshold
                print("pass!")

                print(f"grad values out: {v1.grad[b][h][s]}")
                print(f"grad values ref: {v2.grad[b][h][s]}")
                assert max_relative_error(
                    v1.grad[b][h][s], v2.grad[b][h][s]) < rel_threshold
                print("pass!")

                print(f"grad beta out: {beta1.grad[b][h][s]}")
                print(f"grad beta ref: {beta2.grad[b][h][s]}")
                assert max_relative_error(
                    beta1.grad[b][h][s], beta2.grad[b][h][s]) < rel_threshold
                print("pass!")

                print(f"grad key out: {k1.grad[b][h][s]}")
                print(f"grad key ref: {k2.grad[b][h][s]}")
                assert max_relative_error(
                    k1.grad[b][h][s], k2.grad[b][h][s]) < rel_threshold
                print("pass!")

    print("All tests pass.")

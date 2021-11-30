# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
# https://github.com/idiap/fast-transformers/blob/master/fast_transformers/causal_product/__init__.py
# Modifications Copyright (c) 2021 Kazuki Irie

import os
import torch
from torch.utils.cpp_extension import load
# Just in time import
# https://pytorch.org/tutorials/advanced/cpp_extens

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'causal_product_cuda.cu')

mod_causal_dot_product_cuda = load(
    name="causal_dot_product", sources=[filename], verbose=True)
mod_causal_dot_backward_cuda = load(
    name="causal_dot_backward", sources=[filename], verbose=True)

causal_dot_product_cuda = mod_causal_dot_product_cuda.causal_dot_product
causal_dot_backward_cuda = mod_causal_dot_backward_cuda.causal_dot_backward


class FastWeightSum(torch.autograd.Function):
    """Fast weights with the sum update rule."""
    dot = {
        # "cpu": causal_dot_product_cpu,
        "cuda": causal_dot_product_cuda
    }
    dot_backward = {
        # "cpu": causal_dot_backward_cpu,
        "cuda": causal_dot_backward_cuda
    }

    @staticmethod
    def forward(ctx, Q, K, V, W):

        # Create the output tensor
        device = Q.device
        N, H, L, E = Q.shape
        _, _, _, M = V.shape

        product = torch.zeros((N, H, L, M), device=device, dtype=Q.dtype)

        # Actually perform the dot product
        FastWeightSum.dot[device.type](
            Q.data,
            K.data,
            V.data,
            W,
            product
        )

        ctx.save_for_backward(Q, K, V)

        return product

    @staticmethod
    def backward(ctx, grad_out):
        # Extract the saved tensors
        Q, K, V = ctx.saved_tensors

        # Allocate memory for the gradients
        grad_Q = torch.zeros_like(Q)
        grad_K = torch.zeros_like(K)
        grad_V = torch.zeros_like(V)

        # Compute the gradients
        FastWeightSum.dot_backward[Q.device.type](
            Q.data,
            K.data,
            V.data,
            grad_out,
            grad_Q,
            grad_K,
            grad_V,
        )

        return grad_Q, grad_K, grad_V, None


# Alias the autograd functions to python style snake case naming
fast_weight_sum = FastWeightSum.apply


if __name__ == '__main__':
    import torch
    # Tests pass if the relative difference compared with
    # the corresponding torch autograd computation
    # is smaller than a threshold.

    # Ideally should be tested with double...
    rel_threshold = 1e-3

    # from https://github.com/idiap/fast-transformers/blob/master/tests/causal_product/test_causal_product_gpu.py
    def max_relative_error(a, b, eps=1e-6):
        return torch.abs((a-b) / (torch.abs(a) + eps)).max().item()

    print('##########################')
    print('# Test forward pass')
    print('##########################')

    bsz, n_head, slen, d_head = 3, 5, 7, 11
    v_dim = 4
    # (B, H, len, dim)
    q0 = torch.rand(3, 5, 7, 11).to(0)
    k0 = torch.rand(3, 5, 7, 11).to(0)
    v0 = torch.rand(3, 5, 7, 4).to(0)

    q0 = q0 / q0.sum(dim=-1, keepdim=True)
    k0 = k0 / k0.sum(dim=-1, keepdim=True)

    q1 = torch.zeros(3, 5, 7, 11, requires_grad=True).to(0)
    k1 = torch.zeros(3, 5, 7, 11, requires_grad=True).to(0)
    v1 = torch.zeros(3, 5, 7, v_dim, requires_grad=True).to(0)
    q1.data = q0.data
    k1.data = k0.data
    v1.data = v0.data

    W1 = torch.zeros(3, 5, 11, v_dim).to(0)
    out1 = fast_weight_sum(q1, k1, v1, W1)

    # compute using torch
    q2 = torch.zeros(3, 5, 7, 11, requires_grad=True).to(0)
    k2 = torch.zeros(3, 5, 7, 11, requires_grad=True).to(0)
    v2 = torch.zeros(3, 5, 7, v_dim, requires_grad=True).to(0)

    q2.data = q0.data
    k2.data = k0.data
    v2.data = v0.data

    # (len, B, H, dim)
    q_2 = q2.permute(2, 0, 1, 3)
    slen, bsz, n_head, d_head = q_2.shape

    q_2 = q_2.reshape(slen, bsz * n_head, d_head)

    k_2 = k2.permute(2, 0, 1, 3)
    k_2 = k_2.reshape(slen, bsz * n_head, d_head)

    v_2 = v2.permute(2, 0, 1, 3)
    v_2 = v_2.reshape(slen, bsz * n_head, v_dim)

    W = torch.zeros(3 * 5, v_dim, 11).to(0)

    out_list = []

    for pos in range(slen):
        W = W + torch.bmm(v_2[pos].unsqueeze(2), k_2[pos].unsqueeze(1))
        out_t = torch.bmm(W, q_2[pos].unsqueeze(2)).squeeze()
        out_list.append(out_t.clone())

    out2 = torch.stack(out_list)
    out2 = out2.view(slen, bsz, n_head, v_dim)
 
    out1 = out1.permute(2, 0, 1, 3)

    for s in range(slen):
        for b in range(bsz):
            for h in range(n_head):
                print(f"out1: {out1[s][b][h]}")
                print(f"out2: {out2[s][b][h]}")
                assert max_relative_error(
                    out1[s][b][h], out2[s][b][h]) < rel_threshold
                print("pass!")

    print('##########################')
    print('# Test Backward pass')
    print('##########################')

    # grad 
    loss1 = out1.sum()
    q1.retain_grad()
    k1.retain_grad()
    v1.retain_grad()

    loss1.backward()

    loss2 = out2.sum()
    q2.retain_grad()
    k2.retain_grad()
    v2.retain_grad()

    loss2.backward()

    for s in range(slen):
        for b in range(bsz):
            for h in range(n_head):
                print(f"s={s}, b={b}, h={h}")
                print(f"grad query1: {q1.grad[b][h][s]}")
                print(f"grad query2: {q2.grad[b][h][s]}")
                assert max_relative_error(
                    q1.grad[b][h][s], q2.grad[b][h][s]) < rel_threshold
                print("pass!")
   
                print(f"grad key1: {k1.grad[b][h][s]}")
                print(f"grad key2: {k2.grad[b][h][s]}")
                assert max_relative_error(
                    k1.grad[b][h][s], k2.grad[b][h][s]) < rel_threshold
                print("pass!")

                print(f"grad values1: {v1.grad[b][h][s]}")
                print(f"grad values2: {v2.grad[b][h][s]}")
                assert max_relative_error(
                    v1.grad[b][h][s], v2.grad[b][h][s]) < rel_threshold
                print("pass!")

    print("All tests pass.")


# Adaptation of the original code from
# https://github.com/idiap/fast-transformers/blob/master/fast_transformers/causal_product/__init__.py
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
# Modifications Copyright (c) 2021 Kazuki Irie

import sys
import os
import torch
import torch.nn.functional as F

from torch.utils.cpp_extension import load
# Just in time import
# https://pytorch.org/tutorials/advanced/cpp_extens

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'self_ref_v0.cu')

fwd_cuda = load(
    extra_cuda_cflags=['--ftemplate-depth=1024'],
    name="self_ref_forward",
    sources=[filename], verbose=True)

bwd_cuda = load(
    extra_cuda_cflags=['--ftemplate-depth=1024'],
    name="self_ref_backward",
    sources=[filename], verbose=True)


self_ref_fwd_cuda = fwd_cuda.self_ref_forward
self_ref_bwd_cuda = bwd_cuda.self_ref_backward


class SelfRefv0(torch.autograd.Function):

    dot = {
        # "cpu": causal_dot_product_cpu,
        "cuda": self_ref_fwd_cuda
    }
    dot_backward = {
        # "cpu": causal_dot_backward_cpu,
        "cuda": self_ref_bwd_cuda
    }

    @staticmethod
    def forward(ctx, x, W_y, W_q, W_k, w_b):

        # Shape of x: (B, len, D)
        # Shape of W_q: (n_head, D, E) where n_head * E = D (typically)
        device = x.device
        N, H, L, E = x.shape

        assert W_y.shape == (N, H, E, E), "Reshape/unsqueeze if needed."
        assert W_q.shape == (N, H, E, E), "Reshape/unsqueeze if needed."
        assert W_k.shape == (N, H, E, E), "Reshape/unsqueeze if needed."
        assert w_b.shape == (N, H, E, 4), "Reshape/unsqueeze if needed."

        out = torch.zeros((N, H, L, E), device=device, dtype=x.dtype)  # y

        q_main = torch.zeros((N, H, L, E), device=device, dtype=x.dtype)
        k_main = torch.zeros((N, H, L, E), device=device, dtype=x.dtype)
        beta_main = torch.zeros((N, H, L, 4), device=device, dtype=x.dtype)

        y_diff = torch.zeros((N, H, L, E), device=device, dtype=x.dtype)
        q_diff = torch.zeros((N, H, L, E), device=device, dtype=x.dtype)
        k_diff = torch.zeros((N, H, L, E), device=device, dtype=x.dtype)
        beta_diff = torch.zeros((N, H, L, 4), device=device, dtype=x.dtype)

        # x = F.softmax(x, dim=-1)  # apply already softmax to input

        SelfRefv0.dot[device.type](
            x,
            W_y,
            W_q,
            W_k,
            w_b,
            q_main,
            k_main,
            beta_main,
            y_diff,
            q_diff,
            k_diff,
            beta_diff,
            out
        )

        ctx.save_for_backward(
            x, q_main, k_main, beta_main, y_diff, q_diff, k_diff, beta_diff,
            W_y, W_q, W_k, w_b)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        # Extract the saved tensors
        (x, q, k, beta, y_diff, q_diff, k_diff, beta_diff,
         W_y, W_q, W_k, w_b) = ctx.saved_tensors

        # Allocate memory for the gradients
        grad_x = torch.zeros_like(x)
        grad_W_y = torch.zeros_like(W_y)
        grad_W_q = torch.zeros_like(W_q)
        grad_W_k = torch.zeros_like(W_k)
        grad_w_b = torch.zeros_like(w_b)

        # out_delayed = torch.tanh(torch.cat([h0, out[:, :, :-1]], dim=2))

        # x, W_y, W_q, W_k, w_b
        # W_y_copy = W_y.detach().clone()
        # W_q_copy = W_q.detach().clone()
        # W_k_copy = W_k.detach().clone()
        # w_b_copy = w_b.detach().clone()

        # Compute the gradients
        SelfRefv0.dot_backward[x.device.type](
            x,
            q,
            k,
            beta,
            y_diff,
            q_diff,
            k_diff,
            beta_diff,
            grad_out,
            # W_y_copy,
            # W_q_copy,
            # W_k_copy,
            # w_b_copy,
            W_y,
            W_q,
            W_k,
            w_b,
            grad_x,
            grad_W_y,
            grad_W_q,
            grad_W_k,
            grad_w_b
        )

        return grad_x, grad_W_y, grad_W_q, grad_W_k, grad_w_b


# Alias the autograd functions to python style snake case naming
self_ref_v0 = SelfRefv0.apply


if __name__ == '__main__':
    import torch
    import torch.nn.functional as F

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

    # bsz, n_head, slen, d_head = 3, 5, 11, 8
    bsz, n_head, slen, d_head = 3, 5, 11, 8
    v_dim = d_head * 3 + 4
    print(f"value dim: {v_dim}")

    # W0 = torch.cuda.FloatTensor(
    #     1, n_head, d_head, v_dim, device='cuda').uniform_(-1., 1.)
    # W0 = W0.repeat(bsz, 1, 1, 1)
    Wy0 = torch.cuda.FloatTensor(
        bsz, n_head, d_head, d_head, device='cuda').uniform_(-1., 1.)
    Wq0 = torch.cuda.FloatTensor(
        bsz, n_head, d_head, d_head, device='cuda').uniform_(-1., 1.)
    Wk0 = torch.cuda.FloatTensor(
        bsz, n_head, d_head, d_head, device='cuda').uniform_(-1., 1.)
    wb0 = torch.cuda.FloatTensor(
        bsz, n_head, d_head, 4, device='cuda').uniform_(-1., 1.)

    x0 = torch.rand(bsz, n_head, slen, d_head, device='cuda')

    W_y1 = torch.zeros(
        bsz, n_head, d_head, d_head, requires_grad=True, device='cuda')
    W_q1 = torch.zeros(
        bsz, n_head, d_head, d_head, requires_grad=True, device='cuda')
    W_k1 = torch.zeros(
        bsz, n_head, d_head, d_head, requires_grad=True, device='cuda')
    w_b1 = torch.zeros(
        bsz, n_head, d_head, 4, requires_grad=True, device='cuda')

    W_y1 = Wy0.detach().clone().requires_grad_(True)
    W_q1 = Wq0.detach().clone().requires_grad_(True)
    W_k1 = Wk0.detach().clone().requires_grad_(True)
    w_b1 = wb0.detach().clone().requires_grad_(True)

    W_y2_slow = torch.zeros(
        bsz, n_head, d_head, d_head, requires_grad=True, device='cuda')
    W_q2_slow = torch.zeros(
        bsz, n_head, d_head, d_head, requires_grad=True, device='cuda')
    W_k2_slow = torch.zeros(
        bsz, n_head, d_head, d_head, requires_grad=True, device='cuda')
    w_b2_slow = torch.zeros(
        bsz, n_head, d_head, 4, requires_grad=True, device='cuda')

    W_y2_slow = Wy0.detach().clone().requires_grad_(True)
    W_q2_slow = Wq0.detach().clone().requires_grad_(True)
    W_k2_slow = Wk0.detach().clone().requires_grad_(True)
    w_b2_slow = wb0.detach().clone().requires_grad_(True)

    x1 = torch.zeros(
        bsz, n_head, slen, d_head, requires_grad=True, device='cuda')
    x1 = x0.detach().clone().requires_grad_(True)

    print("Forwarding custom kernel...")
    # softmax done inside self_ref_v0
    out1 = self_ref_v0(x1, W_y1, W_q1, W_k1, w_b1)
    print("done.")

    x2 = torch.zeros(
        bsz, n_head, slen, d_head, requires_grad=True, device='cuda')
    # apply softmax here
    x2 = F.softmax(x0.detach(), dim=-1).clone().requires_grad_(True)

    x2 = x2.permute(2, 0, 1, 3)  # (len, B, H, dim)
    x2 = x2.reshape(slen, bsz * n_head, d_head)  # (len, B*H, dim)

    # W2 = W2.reshape(bsz * n_head, d_head, v_dim)
    W_y2 = W_y2_slow.view(bsz * n_head, d_head, d_head)
    W_q2 = W_q2_slow.view(bsz * n_head, d_head, d_head)
    W_k2 = W_k2_slow.view(bsz * n_head, d_head, d_head)
    w_b2 = w_b2_slow.view(bsz * n_head, d_head, 4)

    out_list = []

    # out = x2[0]  # (B * H, D)
    print("Forwarding PyTorch code...")
    for pos in range(slen):
        out = x2[pos].unsqueeze(1)
        # out = F.softmax(x2[pos], dim=-1).unsqueeze(1)
        # out: (B * H, 1, D)
        # W2: (B * H, D, v_dim)
        # bmm (b,n,M) x (b,M,p) -> (b,n,p)
        # ykqb before squeeze: (B * H, 1, v_dim)
        y = torch.bmm(out, W_y2).squeeze(1)
        out_t = y.reshape(bsz, n_head, d_head)
        out_list.append(out_t.clone())

        if pos < slen - 1:  # no need to update weights at the last time step
            q = torch.bmm(out, W_q2).squeeze(1)
            k = torch.bmm(out, W_k2).squeeze(1)
            beta = torch.bmm(out, w_b2).squeeze(1)
            beta = torch.sigmoid(beta)
            beta_y, beta_q, beta_k, beta_beta = torch.split(
                beta, [1, 1, 1, 1], dim=-1)

            k = F.softmax(k, dim=-1)
            q = F.softmax(q, dim=-1)

            # retrieve currently stored value
            y_old = torch.bmm(k.unsqueeze(1), W_y2).squeeze(1)
            q_old = torch.bmm(k.unsqueeze(1), W_q2).squeeze(1)
            k_old = torch.bmm(k.unsqueeze(1), W_k2).squeeze(1)
            beta_old = torch.bmm(k.unsqueeze(1), w_b2).squeeze(1)

            y_new = torch.bmm(q.unsqueeze(1), W_y2).squeeze(1)
            q_new = torch.bmm(q.unsqueeze(1), W_q2).squeeze(1)
            k_new = torch.bmm(q.unsqueeze(1), W_k2).squeeze(1)
            beta_new = torch.bmm(q.unsqueeze(1), w_b2).squeeze(1)

            # update all weights
            y_insert = beta_y * (y_new - y_old)
            q_insert = beta_q * (q_new - q_old)
            k_insert = beta_k * (k_new - k_old)
            beta_insert = beta_beta * (beta_new - beta_old)

            W_y2 = W_y2.clone() + torch.bmm(
                k.unsqueeze(2), y_insert.unsqueeze(1))
            W_q2 = W_q2.clone() + torch.bmm(
                k.unsqueeze(2), q_insert.unsqueeze(1))
            W_k2 = W_k2.clone() + torch.bmm(
                k.unsqueeze(2), k_insert.unsqueeze(1))
            w_b2 = w_b2.clone() + torch.bmm(
                k.unsqueeze(2), beta_insert.unsqueeze(1))

    print("done.")

    out2 = torch.stack(out_list)
    out2 = out2.view(slen, bsz, n_head, d_head)

    out1 = out1.permute(2, 0, 1, 3)
    for s in range(slen):
        for b in range(bsz):
            for h in range(n_head):
                print(f"s={s}, b={b}, h={h}")
                print(f"out: {out1[s][b][h]}")
                print(f"ref: {out2[s][b][h]}")
                assert max_relative_error(
                    out1[s][b][h], out2[s][b][h]) < rel_threshold
                print("pass!")
    print("==> Forward pass test done.")

    # sys.exit(0)
    print('##########################')
    print('# Test Backward pass')
    print('##########################')

    # grad
    loss1 = out1.sum()
    W_y1.retain_grad()
    W_q1.retain_grad()
    W_k1.retain_grad()
    w_b1.retain_grad()
    x1.retain_grad()
    loss1.backward()

    loss2 = out2.sum()
    W_y2_slow.retain_grad()
    W_q2_slow.retain_grad()
    W_k2_slow.retain_grad()
    w_b2_slow.retain_grad()
    x2.retain_grad()
    loss2.backward()

    print('##########################')
    print('# Gradients input')
    print('##########################')

    x2_grad = x2.grad.reshape(slen, bsz, n_head, d_head)
    x2_grad = x2_grad.permute(1, 2, 0, 3)

    for s in reversed(range(slen)):
        for b in range(bsz):
            for h in range(n_head):
                print(f"s={s}, b={b}, h={h}")
                print(f"grad x out: {x1.grad[b][h][s]}")
                print(f"grad x ref: {x2_grad[b][h][s]}")
                assert max_relative_error(
                    x1.grad[b][h][s], x2_grad[b][h][s]) < rel_threshold
                print("pass!")

    print('##########################')
    print('# Gradients weights')
    print('##########################')

    W_y2_grad = W_y2_slow.grad.reshape(bsz, n_head, d_head, d_head)
    W_q2_grad = W_q2_slow.grad.reshape(bsz, n_head, d_head, d_head)
    W_k2_grad = W_k2_slow.grad.reshape(bsz, n_head, d_head, d_head)
    w_b2_grad = w_b2_slow.grad.reshape(bsz, n_head, d_head, 4)

    print('##########################')
    print('# Gradient Wy')
    print('##########################')
    for b in range(bsz):
        for h in range(n_head):
            for d in range(d_head):
                print(f"b={b} h={h} d={d} ------------------------")
                print(f"grad Wy out: {W_y1.grad[b][h][d]}")
                print(f"grad Wy ref: {W_y2_grad[b][h][d]}")
                assert max_relative_error(
                    W_y1.grad[b][h][d], W_y2_grad[b][h][d]) < rel_threshold
                print("pass!")

    print('##########################')
    print('# Gradient Wq')
    print('##########################')
    for b in range(bsz):
        for h in range(n_head):
            for d in range(d_head):
                print(f"b={b} h={h} d={d} ------------------------")
                print(f"grad Wq out: {W_q1.grad[b][h][d]}")
                print(f"grad Wq ref: {W_q2_grad[b][h][d]}")
                assert max_relative_error(
                    W_q1.grad[b][h][d], W_q2_grad[b][h][d]) < rel_threshold
                print("pass!")

    print('##########################')
    print('# Gradient Wk')
    print('##########################')
    for b in range(bsz):
        for h in range(n_head):
            for d in range(d_head):
                print(f"b={b} h={h} d={d} ------------------------")
                print(f"grad Wk out: {W_k1.grad[b][h][d]}")
                print(f"grad Wk ref: {W_k2_grad[b][h][d]}")
                assert max_relative_error(
                    W_k1.grad[b][h][d], W_k2_grad[b][h][d]) < rel_threshold
                print("pass!")

    print('##########################')
    print('# Gradient wb')
    print('##########################')
    for b in range(bsz):
        for h in range(n_head):
            for d in range(d_head):
                print(f"b={b} h={h} d={d} ------------------------")
                print(f"grad wb out: {w_b1.grad[b][h][d]}")
                print(f"grad wb ref: {w_b2_grad[b][h][d]}")
                assert max_relative_error(
                    w_b1.grad[b][h][d], w_b2_grad[b][h][d]) < rel_threshold
                print("pass!")

    print("==> All tests pass!")

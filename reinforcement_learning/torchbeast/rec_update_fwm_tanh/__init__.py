# Adaptation of the original code from
# https://github.com/idiap/fast-transformers/blob/master/fast_transformers/causal_product/__init__.py
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
# Modifications Copyright (c) 2021 Kazuki Irie

import os
import torch
from torch.utils.cpp_extension import load
# Just in time import
# https://pytorch.org/tutorials/advanced/cpp_extens

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'rec_update_fwm_tanh.cu')

fwd_cuda = load(
    extra_cuda_cflags=['--ftemplate-depth=1024'],
    name="rec_update_fwm_tanh_forward",
    sources=[filename], verbose=True)
bwd_cuda = load(
    extra_cuda_cflags=['--ftemplate-depth=1024'],
    name="rec_update_fwm_tanh_backward",
    sources=[filename], verbose=True)


causal_dot_product_cuda = fwd_cuda.rec_update_fwm_tanh_forward
causal_dot_backward_cuda = bwd_cuda.rec_update_fwm_tanh_backward


class FastWeightRecUpdateTanh(torch.autograd.Function):
    """Compute the weighted sum of values but attending only to previous
    values."""
    dot = {
        # "cpu": causal_dot_product_cpu,
        "cuda": causal_dot_product_cuda
    }
    dot_backward = {
        # "cpu": causal_dot_backward_cpu,
        "cuda": causal_dot_backward_cuda
    }

    @staticmethod
    def forward(ctx, qx, kx, vx, bx, R_q, R_k, R_v, r_b, W, h0):

        # Shape of x: (B, len, D)
        # Shape of W_q: (n_head, D, E) where n_head * E = D (typically)
        device = qx.device
        N, H, L, E = qx.shape
        _, _, _, M = vx.shape

        assert E == M, "Current kernel only support this case."
        assert R_q.shape == (1, H, M, M), "Reshape/unsqueeze if needed."
        assert R_k.shape == (1, H, M, M), "Reshape/unsqueeze if needed."
        assert R_v.shape == (1, H, M, M), "Reshape/unsqueeze if needed."
        assert r_b.shape == (1, H, 1, M), "Reshape/unsqueeze if needed."

        out = torch.zeros((N, H, L, M), device=device, dtype=qx.dtype)
        v_old = torch.zeros((N, H, L, M), device=device, dtype=qx.dtype)

        q_out = torch.zeros((N, H, L, E), device=device, dtype=qx.dtype)
        k_out = torch.zeros((N, H, L, E), device=device, dtype=qx.dtype)
        v_out = torch.zeros((N, H, L, M), device=device, dtype=qx.dtype)
        beta_out = torch.zeros((N, H, L, 1), device=device, dtype=qx.dtype)

        h_init = h0.detach().clone()  # for backward pass.

        FastWeightRecUpdateTanh.dot[device.type](
            qx.data,
            kx.data,
            vx.data,
            bx.data,
            R_q.data,
            R_k.data,
            R_v.data,
            r_b.data,
            q_out,
            k_out,
            v_out,
            beta_out,
            v_old,
            h0.data,
            W,
            out
        )
        ctx.save_for_backward(out, q_out, k_out, v_out, beta_out, v_old,
                              R_q, R_k, R_v, r_b, W, h_init)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        # Extract the saved tensors
        (out, Q, K, V, beta, V_old,
         R_q, R_k, R_v, r_b, W, h0) = ctx.saved_tensors
        # qx, kx, vx, bx, R_q, R_k, R_v, r_b, W, h0

        # Allocate memory for the gradients
        grad_Q = torch.zeros_like(Q)
        grad_K = torch.zeros_like(K)
        grad_V = torch.zeros_like(V)
        grad_beta = torch.zeros_like(beta)
        # R_q, R_k, R_v, r_b
        grad_R_q = torch.zeros_like(R_q)
        grad_R_k = torch.zeros_like(R_k)
        grad_R_v = torch.zeros_like(R_v)
        grad_r_b = torch.zeros_like(r_b)

        out_delayed = torch.tanh(torch.cat([h0, out[:, :, :-1]], dim=2))

        # Compute the gradients
        FastWeightRecUpdateTanh.dot_backward[out.device.type](
            Q.data,
            K.data,
            V.data,
            beta.data,
            V_old.data,
            out_delayed,
            grad_out,
            R_q.data,
            R_k.data,
            R_v.data,
            r_b.data,
            W.data,
            grad_Q,
            grad_K,
            grad_V,
            grad_beta,
            grad_R_q,
            grad_R_k,
            grad_R_v,
            grad_r_b
        )

        return (grad_Q, grad_K, grad_V, grad_beta,
                grad_R_q, grad_R_k, grad_R_v, grad_r_b, None, None)


# Alias the autograd functions to python style snake case naming
rec_update_fwm_tanh = FastWeightRecUpdateTanh.apply


if __name__ == '__main__':
    import torch
    import torch.nn.functional as F

    torch.manual_seed(111)
    # Tests pass if the relative difference compared with
    # the corresponding torch autograd computation
    # is smaller than a threshold.

    # Ideally should be tested with double...
    rel_threshold = 1e-2
    # rel_threshold = 10

    # from https://github.com/idiap/fast-transformers/blob/master/tests/causal_product/test_causal_product_gpu.py
    def max_relative_error(a, b, eps=1e-6):
        return float(torch.abs((b - a) / (torch.abs(b) + eps)).max().item())

    print('##########################')
    print('# Test forward pass')
    print('##########################')

    bsz, n_head, slen, d_head = 3, 5, 11, 32
    v_dim = d_head
    tot_dim = d_head

    R_q0 = torch.cuda.FloatTensor(
        1, n_head, tot_dim, tot_dim, device='cuda').uniform_(-1., 1.)
    R_k0 = torch.cuda.FloatTensor(
        1, n_head, tot_dim, tot_dim, device='cuda').uniform_(-1., 1.)
    R_v0 = torch.cuda.FloatTensor(
        1, n_head, tot_dim, tot_dim, device='cuda').uniform_(-1., 1.)
    r_b0 = torch.cuda.FloatTensor(
        1, n_head, 1, tot_dim, device='cuda').uniform_(-1., 1.)

    q0 = torch.rand(bsz, n_head, slen, d_head, device='cuda')
    k0 = torch.rand(bsz, n_head, slen, d_head, device='cuda')
    v0 = torch.rand(bsz, n_head, slen, v_dim, device='cuda')
    beta0 = torch.rand(bsz, n_head, slen, 1, device='cuda')
    h0 = torch.zeros(bsz, n_head, 1, v_dim, device='cuda')

    R_q1 = torch.zeros(1, n_head, tot_dim, tot_dim, requires_grad=True,
                       device='cuda')
    R_k1 = torch.zeros(1, n_head, tot_dim, tot_dim, requires_grad=True,
                       device='cuda')
    R_v1 = torch.zeros(1, n_head, tot_dim, tot_dim, requires_grad=True,
                       device='cuda')
    r_b1 = torch.zeros(1, n_head, 1, tot_dim, requires_grad=True,
                       device='cuda')

    R_q1.data = R_q0.data
    R_k1.data = R_k0.data
    R_v1.data = R_v0.data
    r_b1.data = r_b0.data

    q1 = torch.zeros(
        bsz, n_head, slen, d_head, requires_grad=True, device='cuda')
    k1 = torch.zeros(
        bsz, n_head, slen, d_head, requires_grad=True, device='cuda')
    v1 = torch.zeros(
        bsz, n_head, slen, v_dim, requires_grad=True, device='cuda')
    beta1 = torch.zeros(
        bsz, n_head, slen, 1, requires_grad=True, device='cuda')

    q1.data = q0.data
    k1.data = k0.data
    v1.data = v0.data
    beta1.data = beta0.data

    W1 = torch.zeros(bsz, n_head, d_head, v_dim, device='cuda')
    print("Forwarding custom kernel...")
    out1 = rec_update_fwm_tanh(
        q1, k1, v1, beta1, R_q1, R_k1, R_v1, r_b1, W1, h0)
    print("done.")

    # Compute it using torch
    Rq2 = torch.zeros(
        1, n_head, tot_dim, tot_dim, requires_grad=True, device='cuda')
    Rk2 = torch.zeros(
        1, n_head, tot_dim, tot_dim, requires_grad=True, device='cuda')
    Rv2 = torch.zeros(
        1, n_head, tot_dim, tot_dim, requires_grad=True, device='cuda')
    rb2 = torch.zeros(
        1, n_head, 1, tot_dim, requires_grad=True, device='cuda')

    Rq2.data = R_q0.data
    Rk2.data = R_k0.data
    Rv2.data = R_v0.data
    rb2.data = r_b0.data

    R_q2 = Rq2.squeeze()
    R_k2 = Rk2.squeeze()
    R_v2 = Rv2.squeeze()
    r_b2 = rb2.squeeze().unsqueeze(2)

    q2 = torch.zeros(
        bsz, n_head, slen, d_head, requires_grad=True, device='cuda')
    k2 = torch.zeros(
        bsz, n_head, slen, d_head, requires_grad=True, device='cuda')
    v2 = torch.zeros(
        bsz, n_head, slen, v_dim, requires_grad=True, device='cuda')
    beta2 = torch.zeros(
        bsz, n_head, slen, 1, requires_grad=True, device='cuda')

    q2.data = q0.data
    k2.data = k0.data
    v2.data = v0.data
    beta2.data = beta0.data

    q_2 = q2.permute(2, 0, 1, 3)
    slen, bsz, n_head, d_head = q_2.shape
    q_2 = q_2.reshape(slen, bsz * n_head, d_head)

    k_2 = k2.permute(2, 0, 1, 3)
    k_2 = k_2.reshape(slen, bsz * n_head, d_head)

    v_2 = v2.permute(2, 0, 1, 3)
    v_2 = v_2.reshape(slen, bsz * n_head, v_dim)

    beta_2 = beta2.permute(2, 0, 1, 3)
    beta_2 = beta_2.reshape(slen, bsz * n_head, 1)

    W = torch.zeros(bsz * n_head, v_dim, d_head, device='cuda')
    out = torch.zeros(n_head, bsz, v_dim, device='cuda')

    out_list = []
    print("Forwarding PyTorch code...")
    for pos in range(slen):
        out = torch.tanh(out)
        # query
        # R_q2 = R_q2.transpose(-1, -2)
        qr = torch.bmm(out, R_q2).squeeze().transpose(0, 1)
        qr = qr.reshape(bsz * n_head, d_head)
        q_t = q_2[pos] + qr
        q_t = F.softmax(q_t, dim=-1)

        # key
        # R_k2 = R_k2.transpose(-1, -2)
        kr = torch.bmm(out, R_k2).squeeze().transpose(0, 1)
        kr = kr.reshape(bsz * n_head, d_head)
        k_t = k_2[pos] + kr
        k_t = F.softmax(k_t, dim=-1)

        # value
        # R_v2 = R_v2.transpose(-1, -2)
        vr = torch.bmm(out, R_v2).squeeze().transpose(0, 1)
        vr = vr.reshape(bsz * n_head, d_head)
        v_t = v_2[pos] + vr

        # beta
        br = torch.bmm(out, r_b2).transpose(0, 1)
        br = br.reshape(bsz * n_head, 1)
        b_t = torch.sigmoid(beta_2[pos] + br)

        # retrieve currently stored value
        v_old = torch.bmm(W, k_t.unsqueeze(2)).squeeze()
        v_insert = b_t * (v_t - v_old)
        W = W + torch.bmm(v_insert.unsqueeze(2), k_t.unsqueeze(1))
        out_t = torch.bmm(W, q_t.unsqueeze(2)).squeeze()
        out_t = out_t.reshape(bsz, n_head, v_dim)
        out = out_t.transpose(0, 1)
        out_list.append(out_t.clone())

    print("done.")

    out2 = torch.stack(out_list)
    out2 = out2.view(slen, bsz, n_head, v_dim)

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

    print('##########################')
    print('# Test Backward pass')
    print('##########################')

    # grad
    loss1 = out1.sum()
    R_q1.retain_grad()
    R_k1.retain_grad()
    R_v1.retain_grad()
    r_b1.retain_grad()
    q1.retain_grad()
    k1.retain_grad()
    v1.retain_grad()
    beta1.retain_grad()
    loss1.backward()

    loss2 = out2.sum()
    R_q2.retain_grad()
    R_k2.retain_grad()
    R_v2.retain_grad()
    r_b2.retain_grad()
    q2.retain_grad()
    k2.retain_grad()
    v2.retain_grad()
    beta2.retain_grad()

    loss2.backward()

    print('##########################')
    print('# Gradients input')
    print('##########################')

    for s in reversed(range(slen)):
        for b in range(bsz):
            for h in range(n_head):
                print(f"s={s}, b={b}, h={h}")
                print(f"grad q out: {q1.grad[b][h][s]}")
                print(f"grad q ref: {q2.grad[b][h][s]}")
                assert max_relative_error(
                    q1.grad[b][h][s], q2.grad[b][h][s]) < rel_threshold
                print("pass!")

                print(f"grad key out: {k1.grad[b][h][s]}")
                print(f"grad key ref: {k2.grad[b][h][s]}")
                assert max_relative_error(
                    k1.grad[b][h][s], k2.grad[b][h][s]) < rel_threshold
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

    print('##########################')
    print('# Gradients weights')
    print('##########################')

    for h in range(n_head):
        print(f"h={h} ------------------------")

        print(f"grad rb out: {r_b1.grad[0][h]}")
        print(f"grad rb ref: {rb2.grad[0][h]}")
        assert max_relative_error(
            r_b1.grad[0][h], rb2.grad[0][h]) < rel_threshold
        print("pass!")

        print(f"grad Rq out: {R_q1.grad[0][h]}")
        print(f"grad Rq ref: {Rq2.grad[0][h]}")
        assert max_relative_error(
            R_q1.grad[0][h], Rq2.grad[0][h]) < rel_threshold
        print("pass!")

        print(f"grad Rk out: {R_k1.grad[0][h]}")
        print(f"grad Rk ref: {Rk2.grad[0][h]}")
        assert max_relative_error(
            R_k1.grad[0][h], Rk2.grad[0][h]) < rel_threshold
        print("pass!")

        print(f"grad Rv out: {R_v1.grad[0][h]}")
        print(f"grad Rv ref: {Rv2.grad[0][h]}")
        assert max_relative_error(
            R_v1.grad[0][h], Rv2.grad[0][h]) < rel_threshold
        print("pass!")

    print("==> All tests pass!")

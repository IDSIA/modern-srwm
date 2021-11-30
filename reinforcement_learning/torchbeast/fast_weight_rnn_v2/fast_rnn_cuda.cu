// Original code from https://github.com/idiap/fast-transformers/blob/master/fast_transformers/causal_product/causal_product_cuda.cu
// Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
// Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
// Apoorv Vyas <avyas@idiap.ch>
//
// Modified to implement the fast RNN V2 with *FWM update rule*.
// V2 applies activation function just before recurrent matmul.
// Copyright (c) 2021 Kazuki Irie

#include <torch/extension.h>
// #include <iostream>
// #include <c10/cuda/CUDAGuard.h>


typedef torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits>
  float_accessor;

// sigmoid
__device__ float sgmf(float x) {
    return 1.f / (1.f + expf(-x));
}


__global__ void fast_rnn_v2_forward_kernel(
    const float_accessor inputs,
    const float_accessor keys,
    const float_accessor values,
    const float_accessor betas,
    float_accessor values_old,
    float_accessor states,
    float_accessor kv,
    float_accessor result,
    float_accessor delayed_norm_res,
    const int N,
    const int H,
    const int L,
    const int E,
    const int M,
    const int E_per_subblock,
    const int subblocks_per_seq,
    const int T,  // block chunk size in time dim.
    const int l_offset  // multiple of T, length offset.
) {
    // Each block takes care of one sequence.
    // blockIdx.x = n * H + h
    int n = blockIdx.x / H;  // batch id
    int h = blockIdx.x % H;  // head id

    // threadIdx.x = e_local*M + m
    // Local e coordinate within E_per_subblock sub-block.
    int e_local = threadIdx.x / M;
    int m = threadIdx.x % M;

    const int E_block = subblocks_per_seq * E_per_subblock;

    // Load the shared memory for KV
    const int shared_kv_size = E_block * M;
    extern __shared__ float shared_mem[];
    float* shared_kv = shared_mem;
    float* shared_results = shared_mem + shared_kv_size;
    float* shared_states = shared_results + M;
    float* shared_values_old = shared_states + M;
    float* shared_betas = shared_values_old + M;
    float* softmax_denom = shared_betas + T;
    float* max_value = softmax_denom + 1;
    float* shared_values = max_value + 1;
    float* shared_keys = shared_values + M*T;
    float* shared_inputs = shared_keys + E_block*T;

    const float esp = 1e-6;

    if (threadIdx.x < M) {
        shared_results[m] = 0.f;
        shared_values_old[m] = 0.f;
    }
    if (threadIdx.x < 1) {
        softmax_denom[0] = 0.f;
        max_value[0] = 0.f;
    }
    // the last segment is shorter.
    int t_end = (T + l_offset) <= L ? T : L - l_offset;

    for (int i = threadIdx.x; i < (t_end*M); i += blockDim.x)
    {
        int t = int(i / M) + l_offset;
        int d = i % M;
        shared_values[i] = values[n][h][t][d];
        shared_inputs[i] = inputs[n][h][t][d];
    }
    for (int i = threadIdx.x; i < (t_end*E_block); i += blockDim.x)
    {
        int t = int(i / E_block) + l_offset;
        int d = (i % E_block);
        if (d < E) {
            shared_keys[i] = keys[n][h][t][d];
        }
    }
    for (int i = threadIdx.x; i < t_end; i += blockDim.x)
    {
        int t = i + l_offset;
        shared_betas[i] = betas[n][h][t][0];
    }
    __syncthreads();
    if (n >= N) {
        return;
    }
    int e;
    int kv_idx;
    for (int sub=0; sub<subblocks_per_seq; sub++) {
        e = sub * E_per_subblock + e_local;
        kv_idx = threadIdx.x + sub * blockDim.x;
        if (e < E) {
            shared_kv[kv_idx] = kv[n][h][e][m];
        }
    }
    // init variables
    if (threadIdx.x < M) {
        // initialize RNN state
        shared_states[m] = states[n][h][0][m];
    }
    int e_abs;
    for (int t=0; t<t_end; t++) {  // loop over time in the segment
        int l = t + l_offset;  // absolute position in time
        int m_abs = t*M + m;

        float v_old;
        for (int sub=0; sub<subblocks_per_seq; sub++) {
            e = sub * E_per_subblock + e_local;
            e_abs = t*E_block + e;
            kv_idx = threadIdx.x + sub * blockDim.x;
            if (e < E) {
                // get old value
                v_old = shared_kv[kv_idx] * shared_keys[e_abs];

                atomicAdd(
                    &shared_values_old[m],
                    v_old
                );
            }
        }
        __syncthreads();
        // For stable softmax
        float max_val;
        float tmp_max;
        if (threadIdx.x < 1) {  // Not parallelized! this should be improved!
            max_val = shared_states[0];
            for (int i = 1; i < M; i++) {
                tmp_max = shared_states[i];
                if (tmp_max > max_val) {
                    max_val = tmp_max;
                }
            }
            max_value[0] = max_val;
        }
        __syncthreads();
        // compute denominator for softmax
        if (threadIdx.x < M) {
            shared_states[m] = expf(shared_states[m] - max_value[0]);
            atomicAdd(
                &softmax_denom[0],
                shared_states[m]
            );
        }
        __syncthreads();
        float res;
        // compute new value to be inserted
        float v_insert = shared_betas[t] * 
          (shared_values[m_abs] - shared_values_old[m]);
        for (int sub=0; sub<subblocks_per_seq; sub++) {
            e = sub * E_per_subblock + e_local;
            e_abs = t*E_block + e;
            kv_idx = threadIdx.x + sub * blockDim.x;
            if (e < E) {
                // Update fast weights
                shared_kv[kv_idx] += shared_keys[e_abs] * v_insert;
                float out_nzd =
                  shared_states[e] / (softmax_denom[0] + esp);
                res = out_nzd * shared_kv[kv_idx];
                atomicAdd(
                    &shared_results[m],  // recurrent part
                    res
                );
            }
        }
        __syncthreads();

        if (threadIdx.x < M) {
            // m = threadIdx.x if threadIdx.x < M
            // Save for backward pass (recomputed...)
            float tmp = shared_states[m] / (softmax_denom[0] + esp);
            atomicAdd(
                &delayed_norm_res[n][h][l][m],
                tmp
            );
            float r1 = shared_results[m] + shared_inputs[m_abs];
            atomicAdd(
                &result[n][h][l][m],
                r1
            );
            shared_states[m] = r1;  // state update
            shared_results[m] = 0.f;

            // same for v_old
            float r2 = shared_values_old[m];
            atomicAdd(
                &values_old[n][h][l][m],
                r2
            );
            shared_values_old[m] = 0.f;
        }
        if (threadIdx.x < 1) {
            softmax_denom[0] = 0.f;
        }
        __syncthreads();
    }
    __syncthreads();
    // write back to kv to be carried over to the next segment.
    for (int sub=0; sub<subblocks_per_seq; sub++) {
        e = sub * E_per_subblock + e_local;
        kv_idx = threadIdx.x + sub * blockDim.x;
        if (e < E) {
            kv[n][h][e][m] = shared_kv[kv_idx];
        }
    }
    if (threadIdx.x < M) {
        states[n][h][0][m] = shared_states[m];
    }
}


// Forward
void fast_rnn_v2_forward(
    const torch::Tensor inputs,
    const torch::Tensor keys,
    const torch::Tensor values,
    const torch::Tensor betas,
    torch::Tensor v_old,
    torch::Tensor states,  // init states
    torch::Tensor kv,  // might be non zero if carried over from previous seg.
    torch::Tensor outputs,
    torch::Tensor out_de_nmz
) {
//    const at::cuda::OptionalCUDAGuard device_guard(device_of(queries));
    int N = inputs.size(0);
    int H = inputs.size(1);
    int L = inputs.size(2);
    int E = inputs.size(3);
    int M = values.size(3);

    int threads = 1024;

    // Shared mem max size is 48KB
    int MUL_PER_BLOCK = min(threads, E*M);
    // make sure that MUL_PER_BLOCK is divisible by M;
    MUL_PER_BLOCK = int(MUL_PER_BLOCK / M) *  M;
    threads = MUL_PER_BLOCK;
    // const int blocks_per_sequence = ((E*M) + threads -1) / threads;
    const int subblocks_per_seq = ((E*M) + threads -1) / threads;

    const int E_per_subblock = MUL_PER_BLOCK / M;
    // int blocks  = N*H*blocks_per_sequence;
    int blocks = N*H;  // total number of sequences
    // KV, +1 output/results, + 1 for states, + 1 old val.
    int shared_mem_const = (subblocks_per_seq * E_per_subblock + 3)*M + 1 + 1;
    // M for value and input, E for key and 1 for beta.
    int shared_mem_per_time = 2*M + E_per_subblock * subblocks_per_seq + 1;

    // T = max time chunk size we can afford
    // 12 * 1024 * 4 (float) = 49KB
    assert(12 * 1024 - shared_mem_const > 0 &&
        "`d_head` too large. To obtain large models, keep `d_head` small"
        "e.g. 64 and increase the number of heads instead.");
    const int T = int(((12 * 1024) - shared_mem_const) / shared_mem_per_time);
    const int shared_mem_forward =
      ((T*shared_mem_per_time) + shared_mem_const) * sizeof(float);

    for (int l_offset=0; l_offset < L; l_offset += T) {
     fast_rnn_v2_forward_kernel
            <<<blocks, MUL_PER_BLOCK, shared_mem_forward>>>(
            inputs.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            keys.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            values.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            betas.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            v_old.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            states.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            kv.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            outputs.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            out_de_nmz.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            N, H, L, E, M, E_per_subblock, subblocks_per_seq, T, l_offset
        );
    }
}


// Backward kernel
__global__ void fast_rnn_v2_backward_kernel(
    const float_accessor keys,
    const float_accessor values,
    const float_accessor betas,
    const float_accessor v_old,
    const float_accessor rnn_out,
    const float_accessor rnn_out_delayed,
    const float_accessor grad_out,
    float_accessor tmp_grad,
    float_accessor kv,
    float_accessor grad_kv,
    float_accessor grad_inputs,
    float_accessor grad_keys,
    float_accessor grad_values,
    float_accessor grad_betas,
    int N,
    int H,
    int L,
    int E,
    int M,
    int E_per_subblock,
    int subblocks_per_seq,
    int T,
    int l_offset
) {
    // Each block takes care of one sequence.
    // blockIdx.x = n * H + h
    int n = blockIdx.x / H;
    int h = blockIdx.x % H;

    // threadIdx.x = e_local*M + m
    // Local e coordinate within E_per_subblock sub-block.
    int e_local = threadIdx.x / M;
    int m = threadIdx.x % M;

    const int E_block = subblocks_per_seq * E_per_subblock;

    // Load the shared memory for KV
    const int shared_kv_size = E_block * M;
    extern __shared__ float shared_mem[];
    float* shared_kv = shared_mem;
    float* shared_grad_kv = shared_mem + shared_kv_size;
    float* shared_res_i = shared_grad_kv + shared_kv_size;
    float* shared_res_k = shared_res_i + M;
    float* shared_res_v = shared_res_k + M;
    float* shared_grad_v_old =shared_res_v + M;
    float* shared_res_beta = shared_grad_v_old + M;
    float* grad_sft_cst = shared_res_beta + 1;

    float* shared_gradout = grad_sft_cst + 1;
    float* shared_keys = shared_gradout + M*T;
    float* shared_values = shared_keys + E_block*T;
    float* shared_rnn_out = shared_values + M*T;
    float* shared_rnn_out_delayed = shared_rnn_out + M*T;
    float* shared_v_old = shared_rnn_out_delayed + M*T;
    float* shared_betas = shared_v_old + M*T;
    float* shared_tmp_grad = shared_betas + T;

    if (threadIdx.x < M) {
        shared_res_i[m] = 0.f;
        shared_res_k[m] = 0.f;
        shared_res_v[m] = 0.f;
        shared_grad_v_old[m] = 0.f;
    }
    if (threadIdx.x < 1) {
        shared_res_beta[0] = 0.f;
        grad_sft_cst[0] = 0.f;  // offset for grad softmax
    }
    // Everythig goes backward
    int t_end = (T + l_offset) <= L ? T : L - l_offset;
    for (int i = threadIdx.x; i < (t_end*M); i += blockDim.x)
    {
        int t = int(i / M) + l_offset;
        int t_bw = L - 1 - t;
        int d = i % M;
        shared_gradout[i] = grad_out[n][h][t_bw][d];
        shared_rnn_out[i] = rnn_out[n][h][t_bw][d];
        shared_values[i] = values[n][h][t_bw][d];
        shared_v_old[i] = v_old[n][h][t_bw][d];
    }
    for (int i = threadIdx.x; i < (t_end*E_block); i += blockDim.x)
    {
        int t = int(i / E_block) + l_offset;
        int t_bw = L - 1 - t;
        int d = (i % E_block);
        if (d < E) {
            shared_rnn_out_delayed[i] = rnn_out_delayed[n][h][t_bw][d];
            shared_keys[i] = keys[n][h][t_bw][d];
        }
    }
    for (int i = threadIdx.x; i < t_end; i += blockDim.x)
    {
        int t = i + l_offset;
        int t_bw = L - 1 - t;
        shared_betas[i] = betas[n][h][t_bw][0];
    }
    __syncthreads();
    if (n >= N) {
        return;
    }
    int e;
    int e_abs;  // absolute idx from t=0
    int kv_idx;
    for (int sub=0; sub<subblocks_per_seq; sub++) {
        e = sub * E_per_subblock + e_local;
        kv_idx = threadIdx.x + sub * blockDim.x;
        if (e < E) {
            shared_kv[kv_idx] = kv[n][h][e][m];
            shared_grad_kv[kv_idx] = grad_kv[n][h][e][m];
        }
    }
    if (threadIdx.x < M) {
        // threadIdx.x = m if threadIdx.x < M
        shared_tmp_grad[m] = tmp_grad[n][h][0][m];
    }

    for (int t=0; t<t_end; t++) {
        int l = t + l_offset;
        int l_b = L - l -1;
        int m_abs = t*M + m;

        if (threadIdx.x < M) {  // element-wise ops only here
            // threadIdx.x = m if threadIdx.x < M
            shared_tmp_grad[m] += shared_gradout[m_abs];
            // add new grad to tmp grad accumulator
            // shared_tmp_grad[m] += shared_gradout[m_abs];
            // __syncthreads();
            // sigmoid
            // float grad_z = 
            //   (1. - shared_rnn_out[m_abs]) * shared_rnn_out[m_abs]
            //   * shared_tmp_grad[m];
            float grad_z = shared_tmp_grad[m];
            // float grad_z = shared_rnn_out[m_abs] * (
            //   shared_tmp_grad[m] - grad_sft_cst[0]);
            // tanh --> grad_z = grad_h * (1- out^2)
            // float grad_z = (
            //   1. - shared_rnn_out[m_abs] * shared_rnn_out[m_abs])
            //   * shared_tmp_grad[m];
            atomicAdd(
                &shared_res_i[m],
                grad_z
            );  // grad for input
            shared_tmp_grad[m] = 0.f;  // prepare grad for the next step.
        }
        __syncthreads();  // important to sync

        float v_diff = shared_values[m_abs] - shared_v_old[m_abs];
        float v_ins = v_diff * shared_betas[t];

        for (int sub=0; sub<subblocks_per_seq; sub++) {
            e = sub * E_per_subblock + e_local;
            e_abs = t*E_block + e;
            kv_idx = threadIdx.x + sub * blockDim.x;
            if (e < E) {
                // grad rec weight part 1
                shared_grad_kv[kv_idx] +=
                  shared_res_i[m] * shared_rnn_out_delayed[e_abs];

                // grad v
                float res = shared_keys[e_abs] * shared_grad_kv[kv_idx] 
                  * shared_betas[t];
                atomicAdd(
                    &shared_res_v[m],
                    res
                );

                // grad k part 1 and 2
                float res_k = shared_grad_kv[kv_idx] * v_ins;
                atomicAdd(
                    &shared_res_k[e],
                    res_k
                );

                // grad beta
                float res_beta = shared_grad_kv[kv_idx] * shared_keys[e_abs]
                  * v_diff;
                atomicAdd(
                    &shared_res_beta[0],
                    res_beta
                );

                float res_h = shared_res_i[m] * shared_kv[kv_idx];
                atomicAdd(
                    &shared_tmp_grad[e],
                    res_h
                );
            }
        }
        __syncthreads();
        // compute constant for grad softmax
        if (threadIdx.x < M) {
            float cst = 
              shared_tmp_grad[m] * shared_rnn_out_delayed[m_abs];
            atomicAdd(
                &grad_sft_cst[0],
                cst
            );
        }
        __syncthreads();
        if (threadIdx.x < M) {
            shared_tmp_grad[m] = shared_rnn_out_delayed[m_abs]
              * (shared_tmp_grad[m] - grad_sft_cst[0]);
        }
        if (threadIdx.x < 1) { 
            grad_sft_cst[0] = 0.f;
        }
        for (int sub=0; sub<subblocks_per_seq; sub++) {
            e = sub * E_per_subblock + e_local;
            e_abs = t*E_block + e;
            kv_idx = threadIdx.x + sub * blockDim.x;
            if (e < E) {
                // reverse update kv
                shared_kv[kv_idx] -= shared_keys[e_abs] * v_ins;
                // grad v_old
                float res_v_old = - (shared_grad_kv[kv_idx] * shared_betas[t]
                  * shared_keys[e_abs]);
                atomicAdd(
                  &shared_grad_v_old[m],
                  res_v_old
                );
            }
        }
        __syncthreads();
        // remaining key grad
        for (int sub=0; sub<subblocks_per_seq; sub++) {
            e = sub * E_per_subblock + e_local;
            e_abs = t*E_block + e;
            kv_idx = threadIdx.x + sub * blockDim.x;
            if (e < E) {
                float res_kp3 = shared_grad_v_old[m] * shared_kv[kv_idx];
                atomicAdd(
                    &shared_res_k[e],
                    res_kp3
                );  // remaining key grad
                // grad kv via v old
                shared_grad_kv[kv_idx] +=
                  shared_grad_v_old[m] * shared_keys[e_abs];
            }
        }
        __syncthreads();

        if (threadIdx.x < M) {
            // m = threadIdx.x if threadIdx.x < M
            float ri = shared_res_i[m];
            atomicAdd(
                &grad_inputs[n][h][l_b][m],
                ri
            );

            float rk = shared_res_k[m];
            atomicAdd(
                &grad_keys[n][h][l_b][m],
                rk
            );

            float rv = shared_res_v[m];
            atomicAdd(
                &grad_values[n][h][l_b][m],
                rv
            );
            shared_res_i[m] = 0.f;
            shared_res_k[m] = 0.f;
            shared_res_v[m] = 0.f;
            shared_grad_v_old[m] = 0.f;
        }
        __syncthreads();
        if (threadIdx.x < 1) {
            float r3 = shared_res_beta[0];
            atomicAdd(
                &grad_betas[n][h][l_b][0],
                r3
            );
            shared_res_beta[0] = 0.f;
        }
        __syncthreads();
    }

    __syncthreads();
    // write back temporal gradients.
    for (int sub=0; sub<subblocks_per_seq; sub++) {
        e = sub * E_per_subblock + e_local;
        kv_idx = threadIdx.x + sub * blockDim.x;
        if (e < E) {
            kv[n][h][e][m] = shared_kv[kv_idx];
            grad_kv[n][h][e][m] = shared_grad_kv[kv_idx];
        }
    }
    if (threadIdx.x < M) {
        // threadIdx.x = m if threadIdx.x < M
        tmp_grad[n][h][0][m] = shared_tmp_grad[m];
    }
}


// Backward
void fast_rnn_v2_backward(
    const torch::Tensor keys,
    const torch::Tensor values,
    const torch::Tensor betas,
    const torch::Tensor v_old,
    const torch::Tensor grad_out,
    const torch::Tensor outputs,
    const torch::Tensor o_delayed,
    torch::Tensor fw_mem,  // from the forward pass.
    torch::Tensor grad_in,
    torch::Tensor grad_k,
    torch::Tensor grad_v,
    torch::Tensor grad_beta
) {
//    const at::cuda::OptionalCUDAGuard device_guard(device_of(grad_queries));
    int N = keys.size(0);
    int H = keys.size(1);
    int L = keys.size(2);
    int E = keys.size(3);
    int M = values.size(3);

    auto grad_kv = torch::zeros({N, H, E, M}, keys.options());
    auto tmp_grad = torch::zeros({N, H, 1, M}, keys.options());

    const int threads = 512;

    // First part ====================================
    int MPB = min(threads, E*M);
    // make sure that MUL_PER_BLOCK is divisible by M;
    MPB = int(MPB / M) *  M;
    const int subblocks_per_seq_value = ((E*M) + MPB - 1)/ MPB;
    const int E_per_subblock = MPB / M;
    const int blocks_value = N*H;
    const int E_block = E_per_subblock * subblocks_per_seq_value;

    // 2*E*M for KV and grad_KV, and 3*M.
    int shared_mem_const = (2 * E_block + 4)*M + 2;
    // 5M for value, rnn_out, rnn_delayed, grad_out, tmp_grad.
    // E for key.
    int shared_mem_per_time = 6*M + E_block + 1;
    assert(12 * 1024 - shared_mem_const > 0 &&
           "`d_head` too large. To obtain large models, keep `d_head` small"
           "e.g. 64 and increase the number of heads instead.");
    int T = int(((12 * 1024) - shared_mem_const) / shared_mem_per_time);
    const int shared_mem_v_backward =
      ((T*shared_mem_per_time) + shared_mem_const) * sizeof(float);

    for (int l_offset=0; l_offset < L; l_offset += T) {
        fast_rnn_v2_backward_kernel
            <<<blocks_value, MPB, shared_mem_v_backward>>>(
            keys.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            values.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            betas.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            v_old.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            outputs.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            o_delayed.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            grad_out.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            tmp_grad.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            fw_mem.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            grad_kv.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            grad_in.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            grad_k.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            grad_v.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            grad_beta.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            N, H, L, E, M, E_per_subblock, subblocks_per_seq_value, T, l_offset
        );
    }
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "fast_rnn_v2_forward",
        &fast_rnn_v2_forward,
        "Compute the weighted sum of values but attending only to previous "
        "values."
    );
    m.def(
        "fast_rnn_v2_backward",
        &fast_rnn_v2_backward,
        "Compute the gradients for the fast weight memory."
    );
}

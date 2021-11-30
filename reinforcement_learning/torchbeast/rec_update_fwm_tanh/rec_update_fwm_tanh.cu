// Original code from https://github.com/idiap/fast-transformers/blob/master/fast_transformers/causal_product/causal_product_cuda.cu
// Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
// Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
// Apoorv Vyas <avyas@idiap.ch>
//
// Modified to implement fast weight memory with recurrent update rule;
// where the production of key/value/query/beta at each time step is
// conditioned on the previous output of the layer.
// Copyright (c) 2021 Kazuki Irie

#include <torch/extension.h>
// #include <c10/cuda/CUDAGuard.h>


typedef torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits>
  float_accessor;


// sigmoid
__device__ float sgmf(float x) {
    return 1.f / (1.f + expf(-x));
}


__global__ void rec_update_fwm_tanh_forward_kernel(
    const float_accessor qx,
    const float_accessor kx,
    const float_accessor vx,
    const float_accessor bx,
    const float_accessor Rq,
    const float_accessor Rk,
    const float_accessor Rv,
    const float_accessor rb,
    float_accessor res_queries,
    float_accessor res_keys,
    float_accessor res_values,
    float_accessor res_betas,
    float_accessor values_old,
    float_accessor states,
    float_accessor kv,
    float_accessor result,
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
    int e_local = threadIdx.x / M;  // e index within the sub-block
    int m = threadIdx.x % M;

    const int E_block = subblocks_per_seq * E_per_subblock;

    // Load the shared memory for KV
    const int shared_kv_size = E_block * M;
    extern __shared__ float shared_mem[];
    float* shared_kv = shared_mem;
    float* shared_Rq = shared_mem + shared_kv_size;
    float* shared_Rk = shared_Rq + shared_kv_size;
    float* shared_Rv = shared_Rk + shared_kv_size;
    float* shared_rb = shared_Rv + shared_kv_size;

    float* shared_results = shared_rb + M;
    float* shared_vx = shared_results + M;
    float* shared_kx = shared_vx + M*T;
    float* shared_qx = shared_kx + E_block*T;
    float* shared_v_old = shared_qx + E_block*T;
    float* shared_v_insert = shared_v_old + M;
    float* shared_bx = shared_v_insert + M;

    float* shared_states = shared_bx + T;
    float* softmax_denom_q = shared_states + M;
    float* max_value_q = softmax_denom_q + 1;
    float* softmax_denom_k = max_value_q + 1;
    float* max_value_k = softmax_denom_k + 1;

    float* shared_qr = max_value_k + 1;
    float* shared_kr = shared_qr + M;
    float* shared_vr = shared_kr + M;
    float* shared_br = shared_vr + M;

    if (threadIdx.x < M) {
        shared_results[threadIdx.x] = 0.f;
        shared_v_old[threadIdx.x] = 0.f;
        shared_v_insert[threadIdx.x] = 0.f;

        shared_qr[m] = 0.f;
        shared_kr[m] = 0.f;
        shared_vr[m] = 0.f;
    }
    if (threadIdx.x < 1) {
        // for softmax on q and k
        softmax_denom_q[0] = 0.f;
        max_value_q[0] = 0.f;
        softmax_denom_k[0] = 0.f;
        max_value_k[0] = 0.f;

        shared_br[0] = 0.f;
    }
    // the last segment is shorter.
    int t_end = (T + l_offset) <= L ? T : L - l_offset;

    for (int i = threadIdx.x; i < (t_end*M); i += blockDim.x)
    {
        int t = int(i / M) + l_offset;
        int d = i % M;
        shared_vx[i] = vx[n][h][t][d];
        shared_kx[i] = kx[n][h][t][d];
        shared_qx[i] = qx[n][h][t][d];
    }
    for (int i = threadIdx.x; i < t_end; i += blockDim.x)
    {
        int t = i + l_offset;
        shared_bx[i] = bx[n][h][t][0];
    }
    __syncthreads();
    if (n >= N) {
        return;
    }

    int e;
    // int e_abs;  // absolute idx from t=0
    int kv_idx;

    for (int sub=0; sub<subblocks_per_seq; sub++) {
        e = sub * E_per_subblock + e_local;
        kv_idx = threadIdx.x + sub * blockDim.x;
        if (e < E) {
            shared_kv[kv_idx] = kv[n][h][e][m];
            shared_Rq[kv_idx] = Rq[0][h][e][m];
            shared_Rk[kv_idx] = Rk[0][h][e][m];
            shared_Rv[kv_idx] = Rv[0][h][e][m];
        }
    }
    if (threadIdx.x < M) {
        shared_states[m] = states[n][h][0][m];  // initialize recurrent state.
        shared_rb[m] = rb[0][h][0][m];
    }
    __syncthreads();

    for (int t=0; t<t_end; t++) {  // main loop over time in the segment
        int l = t + l_offset;  // absolute position in time
        int m_abs = t*M + m;

        if (threadIdx.x < M) {
            // tanh
            shared_states[m] = tanhf(shared_states[m]);
        }
        __syncthreads();

        float q_rec, k_rec, v_rec;
        for (int sub=0; sub<subblocks_per_seq; sub++) {
            e = sub * E_per_subblock + e_local;
            // e_abs = t*E_block + e;
            kv_idx = threadIdx.x + sub * blockDim.x;
            if (e < E) {
                // get recurrent parts
                q_rec = shared_Rq[kv_idx] * shared_states[e];
                k_rec = shared_Rk[kv_idx] * shared_states[e];
                v_rec = shared_Rv[kv_idx] * shared_states[e];
                atomicAdd(
                    &shared_qr[m],
                    q_rec
                );
                atomicAdd(
                    &shared_kr[m],
                    k_rec
                );
                atomicAdd(
                    &shared_vr[m],
                    v_rec
                );
            }
        }        
        float b_rec;
        if (threadIdx.x < M) {
            // for beta
            b_rec = shared_rb[m] * shared_states[m];
            atomicAdd(
                &shared_br[0],  // recurrent part of beta
                b_rec
            );
        }
        __syncthreads();
        float max_val_q, max_val_k;
        float tmp_max_q, tmp_max_k;
        if (threadIdx.x < 1) {
            // beta
            shared_br[0] = sgmf(shared_br[0] + shared_bx[t]);

            // find max for stable softmax
            max_val_q = shared_qr[0] + shared_qx[t*M];
            max_val_k = shared_kr[0] + shared_kx[t*M];
            for (int i = 1; i < M; i++) {
                tmp_max_q = shared_qr[i] + shared_qx[t*M + i];
                tmp_max_k = shared_kr[i] + shared_kx[t*M + i];
                if (tmp_max_q > max_val_q) {
                    max_val_q = tmp_max_q;
                }
                if (tmp_max_k > max_val_k) {
                    max_val_k = tmp_max_k;
                }
            }
            max_value_q[0] = max_val_q;
            max_value_k[0] = max_val_k;
        }
        __syncthreads();
        float rq = expf(shared_qr[m] + shared_qx[m_abs] - max_value_q[0]);
        float rk = expf(shared_kr[m] + shared_kx[m_abs] - max_value_k[0]);
        if (threadIdx.x < M) {
            atomicAdd(
                &softmax_denom_q[0],
                rq
            );
            atomicAdd(
                &softmax_denom_k[0],
                rk
            );
        }
        __syncthreads();
        if (threadIdx.x < M) {
            shared_qr[m] = rq / softmax_denom_q[0];  // stable?
            shared_kr[m] = rk / softmax_denom_k[0];
            shared_vr[m] = shared_vr[m] + shared_vx[m_abs];
        }
        __syncthreads();  // Now all query, key, value, beta are ready.

        float v_old;
        for (int sub=0; sub<subblocks_per_seq; sub++) {
            e = sub * E_per_subblock + e_local;
            // e_abs = t*E_block + e;
            kv_idx = threadIdx.x + sub * blockDim.x;
            if (e < E) {
                // get old value
                v_old = shared_kv[kv_idx] * shared_kr[e];

                atomicAdd(
                    &shared_v_old[m],
                    v_old
                );
            }
        }
        __syncthreads();

        // compute new value to be inserted
        if (threadIdx.x < M) {
            // m = threadIdx.x if threadIdx.x < M
            shared_v_insert[m] =
              shared_br[0] * (shared_vr[m] - shared_v_old[m]);
        }
        __syncthreads();

        float res;
        for (int sub=0; sub<subblocks_per_seq; sub++) {
            // Update fast weights
            e = sub * E_per_subblock + e_local;
            // e_abs = t*E_block + e;
            kv_idx = threadIdx.x + sub * blockDim.x;
            if (e < E) {
                shared_kv[kv_idx] += shared_kr[e] * shared_v_insert[m];
                res = shared_qr[e] * shared_kv[kv_idx];
                atomicAdd(
                    &shared_results[m],
                    res
                );
            }
        }
        __syncthreads();

        if (threadIdx.x < M) {
            // m = threadIdx.x if threadIdx.x < M
            float r1 = shared_results[m];
            atomicAdd(
                &result[n][h][l][m],
                r1
            );
            shared_states[m] = r1;
            shared_results[m] = 0.0;

            // same for v_old and v_insert
            float r2 = shared_v_old[m];
            atomicAdd(
                &values_old[n][h][l][m],
                r2
            );
            shared_v_old[m] = 0.0;
            shared_v_insert[m] = 0.0;

            // Also q, k, v, and beta
            float res_q = shared_qr[m];
            atomicAdd(
                &res_queries[n][h][l][m],
                res_q
            );
            shared_qr[m] = 0.f;

            float res_k = shared_kr[m];
            atomicAdd(
                &res_keys[n][h][l][m],
                res_k
            );
            shared_kr[m] = 0.f;

            float res_v = shared_vr[m];
            atomicAdd(
                &res_values[n][h][l][m],
                res_v
            );
            shared_vr[m] = 0.f;
        }
        if (threadIdx.x < 1) {
            float res_beta = shared_br[0];
            atomicAdd(
                &res_betas[n][h][l][0],
                res_beta
            );
            shared_br[0] = 0.f;
            softmax_denom_q[0] = 0.f;
            softmax_denom_k[0] = 0.f;
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
void rec_update_fwm_tanh_forward(
    const torch::Tensor qx,
    const torch::Tensor kx,
    const torch::Tensor vx,
    const torch::Tensor betax,
    const torch::Tensor R_q,
    const torch::Tensor R_k,
    const torch::Tensor R_v,
    const torch::Tensor r_b,
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor beta,
    torch::Tensor v_old,
    torch::Tensor states,
    torch::Tensor kv,  // might be non zero if carried over from previous seg.
    torch::Tensor output
) {
//    const at::cuda::OptionalCUDAGuard device_guard(device_of(queries));
    int N = qx.size(0);  // shape of qx: (B, H, L, E)
    int H = qx.size(1);
    int L = qx.size(2);
    int E = qx.size(3);
    int M = vx.size(3);  // this kernel assumes E = M. TODO make this general.

    int threads = 512;

    // Shared mem max size is 48KB
    int MUL_PER_BLOCK = min(threads, E*M);
    // make sure that MUL_PER_BLOCK is divisible by M;
    MUL_PER_BLOCK = int(MUL_PER_BLOCK / M) *  M;
    threads = MUL_PER_BLOCK;
    // const int blocks_per_sequence = ((E*M) + threads -1) / threads;
    const int subblocks_per_seq = ((E*M) + threads -1) / threads;

    const int E_per_subblock = MUL_PER_BLOCK / M;
    // int blocks  = N*H*blocks_per_sequence;
    const int blocks = N*H;  // total number of sequences
    // KV fast weight, and +1 output/results, + 2 for insert/old value.
    const int E_block = subblocks_per_seq * E_per_subblock;
    const int shared_mem_const = (E_block * 4 + 8)*M + 5;
    // M for value, 2 * E for query and key, 1 for beta.
    const int shared_mem_per_time = M + 2*E_block + 1;

    // Max shared memory size:
    // 12 * 1024 * 4 (float) = 49152 (48KB)
    // for Turing: 65536 (64KB)
    // for Volta: 98304 (96KB)
    int maxB;
    int device_id = 0;
    // int device_id = inputs_i.device();
    // Should to be faster than `cudaGetDeviceProperties` according to: https://developer.nvidia.com/blog/cuda-pro-tip-the-fast-way-to-query-device-properties/
    cudaDeviceGetAttribute(&maxB,
        cudaDevAttrMaxSharedMemoryPerBlockOptin, device_id);
    // std::cout << "Max shared mem: " << maxB << std::endl;
    int maxF = maxB / sizeof(float);
    // Following is needed for sm > 48KB
    cudaFuncSetAttribute(rec_update_fwm_tanh_forward_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize, maxB);
    // cudaDeviceProp prop;
    // cudaGetDeviceProperties(&prop, 0);
    assert(maxF - shared_mem_const > 0 &&
        "`d_head` too large. To obtain large models, keep `d_head` small"
        "e.g. 16 and increase the number of heads instead.");
    // std::cout << "Max shared mem:  " << maxF * sizeof(float) << std::endl;
    // std::cout << "Shared mem const (float): " << 
    //   shared_mem_const * sizeof(float) << std::endl;
    // std::cout << "Remainder: " << maxF - shared_mem_const << std::endl;
    // std::cout << "Shared per time: " << shared_mem_per_time << std::endl;
    const int T = int((maxF - shared_mem_const) / shared_mem_per_time);
    const int shared_mem =
      ((T*shared_mem_per_time) + shared_mem_const) * sizeof(float);

    for (int l_offset=0; l_offset < L; l_offset += T) {
        rec_update_fwm_tanh_forward_kernel
            <<<blocks, MUL_PER_BLOCK, shared_mem>>>(
            qx.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            kx.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            vx.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            betax.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            R_q.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            R_k.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            R_v.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            r_b.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            q.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            k.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            v.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            beta.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            v_old.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            states.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            kv.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            output.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            N, H, L, E, M, E_per_subblock, subblocks_per_seq, T, l_offset
        );
    }
}


// Backward kernel
__global__ void rec_update_fwm_tanh_backward_kernel(
    const float_accessor queries,
    const float_accessor keys,
    const float_accessor values,
    const float_accessor betas,
    const float_accessor v_old,
    const float_accessor out_delayed,
    const float_accessor grad_out,
    const float_accessor Rq,
    const float_accessor Rk,
    const float_accessor Rv,
    const float_accessor rb, 
    float_accessor tmp_grad,
    float_accessor kv,
    float_accessor grad_kv,
    float_accessor grad_q,
    float_accessor grad_k,
    float_accessor grad_v,
    float_accessor grad_betas,
    float_accessor grad_Rq,
    float_accessor grad_Rk,
    float_accessor grad_Rv,
    float_accessor grad_rb,
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
    float* shared_kv = shared_mem;  // E*M
    float* shared_grad_kv = shared_mem + shared_kv_size;  // E*M

    float* shared_Rq = shared_grad_kv + shared_kv_size;  // E*M
    float* shared_Rk = shared_Rq + shared_kv_size;  // E*M
    float* shared_Rv = shared_Rk + shared_kv_size;  // E*M
    float* shared_rb = shared_Rv + shared_kv_size;  // M

    float* shared_grad_q = shared_rb + M;  // M
    float* shared_grad_k = shared_grad_q + M;  // M
    float* shared_grad_v = shared_grad_k + M;  // M
    float* shared_grad_v_old =shared_grad_v + M;  // M
    float* shared_grad_beta = shared_grad_v_old + M;  // 1
    float* grad_sft_cst = shared_grad_beta + 1;  // 1

    float* shared_gradout = grad_sft_cst + 1;  // M*T
    float* shared_q = shared_gradout + M*T;  // E_block*T
    float* shared_k = shared_q + E_block*T;  // E_block*T
    float* shared_v = shared_k + E_block*T;  // M*T
    float* shared_out_delayed = shared_v + M*T;  // M*T
    float* shared_v_old = shared_out_delayed + M*T;  // M*T
    float* shared_betas = shared_v_old + M*T;  // T
    float* shared_tmp_grad = shared_betas + T;  // M*T

    if (threadIdx.x < M) {
        shared_grad_q[m] = 0.f;
        shared_grad_k[m] = 0.f;
        shared_grad_v[m] = 0.f;
        shared_grad_v_old[m] = 0.f;
    }
    if (threadIdx.x < 1) {
        shared_grad_beta[0] = 0.f;
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
        shared_v[i] = values[n][h][t_bw][d];
        shared_v_old[i] = v_old[n][h][t_bw][d];
    }
    for (int i = threadIdx.x; i < (t_end*E_block); i += blockDim.x)
    {
        int t = int(i / E_block) + l_offset;
        int t_bw = L - 1 - t;
        int d = (i % E_block);
        if (d < E) {
            shared_out_delayed[i] = out_delayed[n][h][t_bw][d];
            shared_k[i] = keys[n][h][t_bw][d];
            shared_q[i] = queries[n][h][t_bw][d];
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
            shared_Rq[kv_idx] = Rq[0][h][e][m];
            shared_Rk[kv_idx] = Rk[0][h][e][m];
            shared_Rv[kv_idx] = Rv[0][h][e][m];
        }
    }
    if (threadIdx.x < M) {
        // threadIdx.x = m if threadIdx.x < M
        shared_tmp_grad[m] = tmp_grad[n][h][0][m];
        shared_rb[m] = rb[0][h][0][m];
    }
    __syncthreads();

    // Main loop
    for (int t=0; t<t_end; t++) {
        int l = t + l_offset;
        int l_b = L - l -1;
        int m_abs = t*M + m;

        if (threadIdx.x < M) {  // element-wise ops only here
            // threadIdx.x = m if threadIdx.x < M
            // add new grad to tmp grad accumulator
            shared_tmp_grad[m] += shared_gradout[m_abs];
        }
        __syncthreads();  // important to sync

        for (int sub=0; sub<subblocks_per_seq; sub++) {
            e = sub * E_per_subblock + e_local;
            e_abs = t*E_block + e;
            kv_idx = threadIdx.x + sub * blockDim.x;
            if (e < E) {
                float grad_q = shared_tmp_grad[m] * shared_kv[kv_idx];
                atomicAdd(
                    &shared_grad_q[e],
                    grad_q
                );
            }
        }
        __syncthreads();
        // compute constant for grad softmax
        for (int sub=0; sub<subblocks_per_seq; sub++) {
            e = sub * E_per_subblock + e_local;
            e_abs = t*E_block + e;
            if (e < E && m == 0) {  // only one column
                float cst = shared_grad_q[e] * shared_q[e_abs];
                atomicAdd(
                    &grad_sft_cst[0],
                    cst
                );
            }
        }
        __syncthreads();
        for (int sub=0; sub<subblocks_per_seq; sub++) {
            e = sub * E_per_subblock + e_local;
            e_abs = t*E_block + e;
            if (e < E && m == 0) {  // only one column
                shared_grad_q[e] = shared_q[e_abs] * (
                  shared_grad_q[e] - grad_sft_cst[0]);
            }
        }
        __syncthreads();
        // if (threadIdx.x < M) {
        //     float cst = shared_grad_q[m] * shared_q[m_abs];
        //     atomicAdd(
        //         &grad_sft_cst[0],
        //         cst
        //     );
        // }
        // __syncthreads();
        // if (threadIdx.x < M) {  // element-wise ops only here
        //     shared_grad_q[m] = shared_q[m_abs] * (
        //         shared_grad_q[m] - grad_sft_cst[0]);
        // }  // grad_q done.
        // __syncthreads();  // important to sync
        if (threadIdx.x < 1) { 
            grad_sft_cst[0] = 0.f;
        }
        __syncthreads();

        float v_diff = shared_v[m_abs] - shared_v_old[m_abs];
        float v_ins = v_diff * shared_betas[t];

        for (int sub=0; sub<subblocks_per_seq; sub++) {
            e = sub * E_per_subblock + e_local;
            e_abs = t*E_block + e;
            kv_idx = threadIdx.x + sub * blockDim.x;
            if (e < E) {
                // grad rec weight part 1
                shared_grad_kv[kv_idx] += shared_tmp_grad[m] * shared_q[e_abs];

                // grad v
                float res = shared_k[e_abs] * shared_grad_kv[kv_idx] 
                  * shared_betas[t];
                atomicAdd(
                    &shared_grad_v[m],
                    res
                );  // grad_v done.

                // grad k part 1 and 2
                float res_k = shared_grad_kv[kv_idx] * v_ins;
                atomicAdd(
                    &shared_grad_k[e],
                    res_k
                );

                // grad beta, with sigmoid
                float grad_b = shared_grad_kv[kv_idx] * shared_k[e_abs]
                  * v_diff * shared_betas[t] * (1.f - shared_betas[t]);
                atomicAdd(
                    &shared_grad_beta[0],
                    grad_b
                );  // grad_beta done.
            }
        }
        __syncthreads();

        for (int sub=0; sub<subblocks_per_seq; sub++) {
            e = sub * E_per_subblock + e_local;
            e_abs = t*E_block + e;
            kv_idx = threadIdx.x + sub * blockDim.x;
            if (e < E) {
                // reverse update kv
                shared_kv[kv_idx] -= shared_k[e_abs] * v_ins;
                // grad v_old
                float res_v_old = - (shared_grad_kv[kv_idx] * shared_betas[t]
                  * shared_k[e_abs]);
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
                    &shared_grad_k[e],
                    res_kp3
                );  // remaining key grad
                // grad kv via v old
                shared_grad_kv[kv_idx] +=
                  shared_grad_v_old[m] * shared_k[e_abs];
            }
        }
        __syncthreads();

        // compute constant for grad softmax
        for (int sub=0; sub<subblocks_per_seq; sub++) {
            e = sub * E_per_subblock + e_local;
            e_abs = t*E_block + e;
            if (e < E && m == 0) {  // only one column
                float cst = shared_grad_k[e] * shared_k[e_abs];
                atomicAdd(
                    &grad_sft_cst[0],
                    cst
                );
            }
        }
        __syncthreads();
        for (int sub=0; sub<subblocks_per_seq; sub++) {
            e = sub * E_per_subblock + e_local;
            e_abs = t*E_block + e;
            if (e < E && m == 0) {  // only one column
                shared_grad_k[e] = shared_k[e_abs] * (
                  shared_grad_k[e] - grad_sft_cst[0]);
            }
        }
        __syncthreads();
        if (threadIdx.x < 1) { 
            grad_sft_cst[0] = 0.f;
        }
        __syncthreads();

        // recurrent part
        if (threadIdx.x < M) {
            shared_tmp_grad[m] = 0.f;  // reset for the next time step.
        }
        __syncthreads();
        for (int sub=0; sub<subblocks_per_seq; sub++) {
            e = sub * E_per_subblock + e_local;
            e_abs = t*E_block + e;
            kv_idx = threadIdx.x + sub * blockDim.x;
            if (e < E) {
                float grad_tanh = 1.f - 
                  shared_out_delayed[e_abs] * shared_out_delayed[e_abs];

                float grad_tmp_q = shared_grad_q[m] * shared_Rq[kv_idx];
                atomicAdd(
                    &shared_tmp_grad[e],
                    grad_tmp_q * grad_tanh
                );
                float grad_tmp_k = shared_grad_k[m] * shared_Rk[kv_idx];
                atomicAdd(
                    &shared_tmp_grad[e],
                    grad_tmp_k * grad_tanh
                );
                float grad_tmp_v = shared_grad_v[m] * shared_Rv[kv_idx];
                atomicAdd(
                    &shared_tmp_grad[e],
                    grad_tmp_v * grad_tanh
                );
            }
        }
        __syncthreads();
        for (int sub=0; sub<subblocks_per_seq; sub++) {
            e = sub * E_per_subblock + e_local;
            e_abs = t*E_block + e;
            if (e < E && m == 0) {  // only one column
                float grad_tanh = 1.f - 
                  shared_out_delayed[e_abs] * shared_out_delayed[e_abs];
                shared_tmp_grad[e] += grad_tanh *
                  shared_rb[e] * shared_grad_beta[0];
            }
        }
        __syncthreads();
        // grad for weights
        for (int sub=0; sub<subblocks_per_seq; sub++) {
            e = sub * E_per_subblock + e_local;
            e_abs = t*E_block + e;
            if (e < E) {
                atomicAdd(
                    &grad_Rq[0][h][e][m],
                    shared_grad_q[m] * shared_out_delayed[e_abs]
                );
                atomicAdd(
                    &grad_Rk[0][h][e][m],
                    shared_grad_k[m] * shared_out_delayed[e_abs]
                );
                atomicAdd(
                    &grad_Rv[0][h][e][m],
                    shared_grad_v[m] * shared_out_delayed[e_abs]
                );
                // shared_grad_Rq[kv_idx] =
                //   shared_grad_q[m] * shared_out_delayed[e_abs];
                // shared_grad_Rk[kv_idx] =
                //   shared_grad_k[m] * shared_out_delayed[e_abs];
                // shared_grad_Rv[kv_idx] =
                //   shared_grad_v[m] * shared_out_delayed[e_abs];
            }
            __syncthreads();
        }
        for (int sub=0; sub<subblocks_per_seq; sub++) {
            e = sub * E_per_subblock + e_local;
            e_abs = t*E_block + e;
            if (e < E && m == 0) {  // only one column
                atomicAdd(
                    &grad_rb[0][h][0][e],
                    shared_out_delayed[e_abs] * shared_grad_beta[0]
                );
            }
        }
        __syncthreads();
        if (threadIdx.x < M) {
            // m = threadIdx.x if threadIdx.x < M
            float rq = shared_grad_q[m];
            atomicAdd(
                &grad_q[n][h][l_b][m],
                rq
            );

            float rk = shared_grad_k[m];
            atomicAdd(
                &grad_k[n][h][l_b][m],
                rk
            );

            float rv = shared_grad_v[m];
            atomicAdd(
                &grad_v[n][h][l_b][m],
                rv
            );
            shared_grad_q[m] = 0.f;
            shared_grad_k[m] = 0.f;
            shared_grad_v[m] = 0.f;
            shared_grad_v_old[m] = 0.f;
        }
        __syncthreads();
        if (threadIdx.x < 1) {
            float r3 = shared_grad_beta[0];
            atomicAdd(
                &grad_betas[n][h][l_b][0],
                r3
            );
            shared_grad_beta[0] = 0.f;
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
void rec_update_fwm_tanh_backward(
    const torch::Tensor queries,
    const torch::Tensor keys,
    const torch::Tensor values,
    const torch::Tensor betas,
    const torch::Tensor v_old,
    const torch::Tensor o_delayed,
    const torch::Tensor grad_out,
    const torch::Tensor Rq,
    const torch::Tensor Rk,
    const torch::Tensor Rv,
    const torch::Tensor rb,
    torch::Tensor fw_mem,  // from the forward pass.
    torch::Tensor grad_q,
    torch::Tensor grad_k,
    torch::Tensor grad_v,
    torch::Tensor grad_beta,
    torch::Tensor grad_Rq,
    torch::Tensor grad_Rk,
    torch::Tensor grad_Rv,
    torch::Tensor grad_rb
) {
//    const at::cuda::OptionalCUDAGuard device_guard(device_of(grad_queries));
    int N = keys.size(0);
    int H = keys.size(1);
    int L = keys.size(2);
    int E = keys.size(3);
    int M = values.size(3);

    auto grad_kv = torch::zeros({N, H, E, M}, keys.options());
    auto tmp_grad = torch::zeros({N, H, 1, M}, keys.options());

    const int threads = 384;

    // First part ====================================
    int MPB = min(threads, E*M);
    // make sure that MUL_PER_BLOCK is divisible by M;
    MPB = int(MPB / M) *  M;
    const int subblocks_per_seq_value = ((E*M) + MPB - 1)/ MPB;
    const int E_per_subblock = MPB / M;
    const int blocks = N*H;
    const int E_block = E_per_subblock * subblocks_per_seq_value;

    // 2*E*M for KV and grad_KV, and 3*M.
    int shared_mem_const = (5 * E_block + 5)*M + 2;
    // 5M for value, rnn_out, rnn_delayed, grad_out, tmp_grad.
    // E for key.
    int shared_mem_per_time = 5*M + 2*E_block + 1;

    // Max shared memory size:
    // 12 * 1024 * 4 (float) = 49152 (48KB)
    // for Turing: 65536 (64KB)
    // for Volta: 98304 (96KB)
    int maxB;
    int device_id = 0;
    // int device_id = inputs_i.device();
    // Should to be faster than `cudaGetDeviceProperties` according to: https://developer.nvidia.com/blog/cuda-pro-tip-the-fast-way-to-query-device-properties/
    cudaDeviceGetAttribute(&maxB,
        cudaDevAttrMaxSharedMemoryPerBlockOptin, device_id);
    maxB=49152;
    // std::cout << "Max shared mem: " << maxB << std::endl;
    int maxF = maxB / sizeof(float);
    // Following is needed for sm > 48KB
    cudaFuncSetAttribute(rec_update_fwm_tanh_backward_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize, maxB);
    // cudaDeviceProp prop;
    // cudaGetDeviceProperties(&prop, 0);
    assert(maxF - shared_mem_const > 0 &&
        "`d_head` too large. To obtain large models, keep `d_head` small"
        "e.g. 16 and increase the number of heads instead.");
    // std::cout << "Max shared mem:  " << maxF * sizeof(float) << std::endl;
    // std::cout << "Shared mem const (float): " << 
    //   shared_mem_const * sizeof(float) << std::endl;
    // std::cout << "Remainder: " << maxF - shared_mem_const << std::endl;
    // std::cout << "Shared per time: " << shared_mem_per_time << std::endl;
    const int T = int((maxF - shared_mem_const) / shared_mem_per_time);
    const int shared_mem =
      ((T*shared_mem_per_time) + shared_mem_const) * sizeof(float);

    for (int l_offset=0; l_offset < L; l_offset += T) {
        rec_update_fwm_tanh_backward_kernel
            <<<blocks, MPB, shared_mem>>>(
            queries.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            keys.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            values.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            betas.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            v_old.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            o_delayed.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            grad_out.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            Rq.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            Rk.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            Rv.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            rb.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            tmp_grad.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            fw_mem.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            grad_kv.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            grad_q.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            grad_k.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            grad_v.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            grad_beta.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            grad_Rq.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            grad_Rk.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            grad_Rv.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            grad_rb.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            N, H, L, E, M, E_per_subblock, subblocks_per_seq_value, T, l_offset
        );
    }
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "rec_update_fwm_tanh_forward",
        &rec_update_fwm_tanh_forward,
        "Compute the weighted sum of values but attending only to previous "
        "values."
    );
    m.def(
        "rec_update_fwm_tanh_backward",
        &rec_update_fwm_tanh_backward,
        "Compute the gradients for the fast weight memory."
    );
}

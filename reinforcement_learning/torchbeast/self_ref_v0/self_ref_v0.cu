// Original code from https://github.com/idiap/fast-transformers/blob/master/fast_transformers/causal_product/causal_product_cuda.cu
// Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
// Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
// Apoorv Vyas <avyas@idiap.ch>
//
// Modified to implement self-referential matrix;
// Copyright (c) 2021 Kazuki Irie

#include <torch/extension.h>
#include <iostream>
// #include <c10/cuda/CUDAGuard.h>


typedef torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits>
  float_accessor;


// sigmoid
__device__ float sgmf(float x) {
    return 1.f / (1.f + expf(-x));
}


__global__ void self_ref_forward_kernel(
    const float_accessor x,
    float_accessor Wy,
    float_accessor Wq,
    float_accessor Wk,
    float_accessor wb,
    float_accessor q,
    float_accessor k,
    float_accessor beta,
    float_accessor y_diff,
    float_accessor q_diff,
    float_accessor k_diff,
    float_accessor beta_diff,
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
    float* shared_Wy = shared_mem;
    float* shared_Wq = shared_mem + shared_kv_size;
    float* shared_Wk = shared_Wq + shared_kv_size;
    float* shared_wb = shared_Wk + shared_kv_size;

    float* shared_x = shared_wb + 4 * M;  // using more than needed.
    // float* shared_x = shared_wb + E_block * 4;

    float* shared_results = shared_x + E_block*T;
    float* shared_q_main = shared_results + M;
    float* shared_k_main = shared_q_main + M;
    float* shared_b_main = shared_k_main + M;

    float* shared_y_q = shared_b_main + 4;
    float* shared_q_q = shared_y_q + M;
    float* shared_k_q = shared_q_q + M;
    float* shared_beta_q = shared_k_q + M;

    float* shared_y_k = shared_beta_q + 4;
    float* shared_q_k = shared_y_k + M;
    float* shared_k_k = shared_q_k + M;
    float* shared_beta_k = shared_k_k + M;

    float* shared_y_insert = shared_beta_k + 4;
    float* shared_q_insert = shared_y_insert + M;
    float* shared_k_insert = shared_q_insert + M;
    float* shared_b_insert = shared_k_insert + M;

    float* softmax_denom_q = shared_b_insert + 4;
    float* max_value_q = softmax_denom_q + 1;
    float* softmax_denom_k = max_value_q + 1;
    float* max_value_k = softmax_denom_k + 1;

    if (threadIdx.x < M) {
        shared_results[threadIdx.x] = 0.f;
        shared_q_main[threadIdx.x] = 0.f;
        shared_k_main[threadIdx.x] = 0.f;

        shared_y_insert[threadIdx.x] = 0.f;
        shared_q_insert[threadIdx.x] = 0.f;
        shared_k_insert[threadIdx.x] = 0.f;

        shared_y_q[threadIdx.x] = 0.f;
        shared_q_q[threadIdx.x] = 0.f;
        shared_k_q[threadIdx.x] = 0.f;

        shared_y_k[threadIdx.x] = 0.f;
        shared_q_k[threadIdx.x] = 0.f;
        shared_k_k[threadIdx.x] = 0.f;
    }
    if (threadIdx.x < 4) {
        shared_b_main[m] = 0.f;
        shared_beta_q[m] = 0.f;
        shared_beta_k[m] = 0.f;
        shared_b_insert[m] = 0.f;
    }
    if (threadIdx.x < 1) {
        // for softmax on q and k
        softmax_denom_q[0] = 0.f;
        max_value_q[0] = 0.f;
        softmax_denom_k[0] = 0.f;
        max_value_k[0] = 0.f;
    }
    // the last segment is shorter.
    int t_end = (T + l_offset) <= L ? T : L - l_offset;

    for (int i = threadIdx.x; i < (t_end*E_block);
         i += blockDim.x)
    {
        int t = int(i / E_block) + l_offset;
        int d = (i % E_block);
        if (d < E) {
            shared_x[i] = x[n][h][t][d];
        }
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
            shared_Wy[kv_idx] = Wy[n][h][e][m];
            shared_Wq[kv_idx] = Wq[n][h][e][m];
            shared_Wk[kv_idx] = Wk[n][h][e][m];
            if (m < 4) {
                shared_wb[e + m * E] = wb[n][h][e][m];
            }
        }
    }
    __syncthreads();

    for (int t=0; t<t_end; t++) {  // main loop over time in the segment
        int l = t + l_offset;  // absolute position in time
        // int m_abs = t*M + m;

        float y_main, q_main, k_main, beta_main;
        for (int sub=0; sub<subblocks_per_seq; sub++) {
            e = sub * E_per_subblock + e_local;
            // e_abs = t*E_block + e;
            kv_idx = threadIdx.x + sub * blockDim.x;
            if (e < E) {
                y_main = shared_Wy[kv_idx] * shared_x[t*E_block + e];
                q_main = shared_Wq[kv_idx] * shared_x[t*E_block + e];
                k_main = shared_Wk[kv_idx] * shared_x[t*E_block + e];
                atomicAdd(
                    &shared_results[m],
                    y_main
                );
                atomicAdd(
                    &shared_q_main[m],
                    q_main
                );
                atomicAdd(
                    &shared_k_main[m],
                    k_main
                );
                if (m < 4) {  // beta
                    beta_main = shared_wb[e + m * E] * shared_x[t*E_block + e];
                    atomicAdd(
                        &shared_b_main[m],
                        beta_main
                    );
                }
            }
        }
        __syncthreads();

        if (threadIdx.x < 4) {  // beta
            shared_b_main[m] = sgmf(shared_b_main[m]);
        }

        float max_val_q, max_val_k;
        float tmp_max_q, tmp_max_k;
        if (threadIdx.x < 1) {
            // find max for stable softmax
            max_val_q = shared_q_main[0];
            max_val_k = shared_k_main[0];
            for (int i = 1; i < M; i++) {
                tmp_max_q = shared_q_main[i];
                tmp_max_k = shared_k_main[i];
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

        // apply softmax to query and key
        float rq = expf(shared_q_main[m] - max_value_q[0]);
        float rk = expf(shared_k_main[m] - max_value_k[0]);
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
            shared_q_main[m] = rq / softmax_denom_q[0];  // stable?
            shared_k_main[m] = rk / softmax_denom_k[0];
            // shared_vr[m] = shared_vr[m] + shared_vx[m_abs];
        }
        __syncthreads();  // Now done for the main transformation.

        // ============  end main =============================================
        // Query part, repeat the same op as above using q as input.

        float y_qq, q_qq, k_qq, beta_qq;
        for (int sub=0; sub<subblocks_per_seq; sub++) {
            e = sub * E_per_subblock + e_local;
            // e_abs = t*E_block + e;
            kv_idx = threadIdx.x + sub * blockDim.x;
            if (e < E) {
                // get recurrent parts
                y_qq = shared_Wy[kv_idx] * shared_q_main[e];
                q_qq = shared_Wq[kv_idx] * shared_q_main[e];
                k_qq = shared_Wk[kv_idx] * shared_q_main[e];
                atomicAdd(
                    &shared_y_q[m],
                    y_qq
                );
                atomicAdd(
                    &shared_q_q[m],
                    q_qq
                );
                atomicAdd(
                    &shared_k_q[m],
                    k_qq
                );
                if (m < 4) {  // beta
                    beta_qq = shared_wb[e + m * E] * shared_q_main[e];
                    atomicAdd(
                        &shared_beta_q[m],
                        beta_qq
                    );
                }
            }
        }
        __syncthreads();

        // no need to apply softmax or sigmoid to v_q
        // ================ end query =========================================
        // start key

        float y_kk, q_kk, k_kk, beta_kk;
        for (int sub=0; sub<subblocks_per_seq; sub++) {
            e = sub * E_per_subblock + e_local;
            // e_abs = t*E_block + e;
            kv_idx = threadIdx.x + sub * blockDim.x;
            if (e < E) {
                // get recurrent parts
                y_kk = shared_Wy[kv_idx] * shared_k_main[e];
                q_kk = shared_Wq[kv_idx] * shared_k_main[e];
                k_kk = shared_Wk[kv_idx] * shared_k_main[e];
                atomicAdd(
                    &shared_y_k[m],
                    y_kk
                );
                atomicAdd(
                    &shared_q_k[m],
                    q_kk
                );
                atomicAdd(
                    &shared_k_k[m],
                    k_kk
                );
                if (m < 4) {  // beta
                    beta_kk = shared_wb[e + m * E] * shared_k_main[e];
                    atomicAdd(
                        &shared_beta_k[m],
                        beta_kk
                    );
                }
            }
        }
        __syncthreads();

        // Update Y matrix
        // compute new value to be inserted
        if (threadIdx.x < M) {
            // m = threadIdx.x if threadIdx.x < M
            shared_y_insert[m] =
              shared_b_main[0] * (shared_y_q[m] - shared_y_k[m]);
            shared_q_insert[m] =
              shared_b_main[1] * (shared_q_q[m] - shared_q_k[m]);
            shared_k_insert[m] =
              shared_b_main[2] * (shared_k_q[m] - shared_k_k[m]);
        }
        __syncthreads();

        if (threadIdx.x < 4) {
            // m = threadIdx.x if threadIdx.x < M
            shared_b_insert[m] =
              shared_b_main[3] * (shared_beta_q[m] - shared_beta_k[m]);
        }
        __syncthreads();

        // Update all fast weights
        for (int sub=0; sub<subblocks_per_seq; sub++) {
            // Update fast weights
            // e_abs = t*E_block + e;
            e = sub * E_per_subblock + e_local;
            kv_idx = threadIdx.x + sub * blockDim.x;
            if (e < E) {
                shared_Wy[kv_idx] += shared_k_main[e] * shared_y_insert[m];
                shared_Wq[kv_idx] += shared_k_main[e] * shared_q_insert[m];
                shared_Wk[kv_idx] += shared_k_main[e] * shared_k_insert[m];
                if (m < 4) {  // beta
                    shared_wb[e + m * E] += shared_k_main[e]
                      * shared_b_insert[m];
                }
            }
        }
        __syncthreads();
        // Done, all fast weights updated =====================================
        // Now write back all variables

        if (threadIdx.x < M) {
            // m = threadIdx.x if threadIdx.x < M
            float r1 = shared_results[m];
            atomicAdd(
                &result[n][h][l][m],
                r1
            );
            atomicAdd(
                &q[n][h][l][m],
                shared_q_main[m]
            );
            atomicAdd(
                &k[n][h][l][m],
                shared_k_main[m]
            );
            shared_results[m] = 0.f;
            shared_q_main[m] = 0.f;
            shared_k_main[m] = 0.f;

            atomicAdd(
                &y_diff[n][h][l][m],
                shared_y_q[m] - shared_y_k[m]
            );
            atomicAdd(
                &q_diff[n][h][l][m],
                shared_q_q[m] - shared_q_k[m]
            );
            atomicAdd(
                &k_diff[n][h][l][m],
                shared_k_q[m] - shared_k_k[m]
            );
            shared_y_q[m] = 0.f;
            shared_q_q[m] = 0.f;
            shared_k_q[m] = 0.f;

            shared_y_k[m] = 0.f;
            shared_q_k[m] = 0.f;
            shared_k_k[m] = 0.f;

            shared_y_insert[m] = 0.f;
            shared_q_insert[m] = 0.f;
            shared_k_insert[m] = 0.f;
        }
        if (threadIdx.x < 4) {
            atomicAdd(
                &beta[n][h][l][m],
                shared_b_main[m]
            );
            atomicAdd(
                &beta_diff[n][h][l][m],
                shared_beta_q[m] - shared_beta_k[m]
            );
            shared_b_main[m] = 0.f;
            shared_beta_q[m] = 0.f;
            shared_beta_k[m] = 0.f;
            shared_b_insert[m] = 0.f;
        }
        if (threadIdx.x < 1) {
            softmax_denom_q[0] = 0.f;
            softmax_denom_k[0] = 0.f;
        }
        __syncthreads();
    }
    __syncthreads();

    for (int sub=0; sub<subblocks_per_seq; sub++) {
        e = sub * E_per_subblock + e_local;
        kv_idx = threadIdx.x + sub * blockDim.x;
        if (e < E) {
            Wy[n][h][e][m] = shared_Wy[kv_idx];
            Wq[n][h][e][m] = shared_Wq[kv_idx];
            Wk[n][h][e][m] = shared_Wk[kv_idx];
            if (m < 4) {
                wb[n][h][e][m] = shared_wb[e + m * E];  // only 4 rows for beta
            }
        }
    }
}


// Forward
void self_ref_forward(
    const torch::Tensor x,
    torch::Tensor W_y,
    torch::Tensor W_q,
    torch::Tensor W_k,
    torch::Tensor w_b,
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor beta,
    torch::Tensor y_diff,
    torch::Tensor q_diff,
    torch::Tensor k_diff,
    torch::Tensor beta_diff,
    torch::Tensor output
) {
//    const at::cuda::OptionalCUDAGuard device_guard(device_of(queries));
    const int N = x.size(0);
    const int H = x.size(1);
    const int L = x.size(2);
    const int E = x.size(3);
    const int M = E;  // here M is simply
    // int M = v_old.size(3);  // 3 * E + 4

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
    const int blocks = N*H;  // total number of sequences
    // KV fast weight, and +1 output/results, + 2 for insert/old value.
    const int E_block = subblocks_per_seq * E_per_subblock;
    const int shared_mem_const = E_block*3*M + 4*M + 4 * (3*M + 4) + 4;
    // M for value, 2 * E for query and key, 1 for beta.
    const int shared_mem_per_time = E_block;

    // Max shared memory size:
    // 12 * 1024 * 4 (float) = 49152 (48KB)
    // for Turing: 65536 (64KB)
    // for Volta: 98304 (96KB)
    int maxB;
    // maxB = 49152;
    int device_id = 0;
    // int device_id = inputs_i.device();
    // Should to be faster than `cudaGetDeviceProperties` according to: https://developer.nvidia.com/blog/cuda-pro-tip-the-fast-way-to-query-device-properties/
    cudaDeviceGetAttribute(&maxB,
        cudaDevAttrMaxSharedMemoryPerBlockOptin, device_id);
    // std::cout << "Max shared mem: " << maxB << std::endl;
    int maxF = maxB / sizeof(float);
    // // Following is needed for sm > 48KB
    cudaFuncSetAttribute(self_ref_forward_kernel,
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
    assert(T >= 1 &&
        "`d_head` too large. To obtain large models, keep `d_head` small"
        "e.g. 16 and increase the number of heads instead.");
    const int shared_mem =
      ((T*shared_mem_per_time) + shared_mem_const) * sizeof(float);

    for (int l_offset=0; l_offset < L; l_offset += T) {
        self_ref_forward_kernel
            <<<blocks, MUL_PER_BLOCK, shared_mem>>>(
            x.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            W_y.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            W_q.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            W_k.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            w_b.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            q.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            k.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            beta.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            y_diff.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            q_diff.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            k_diff.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            beta_diff.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            output.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            N, H, L, E, M, E_per_subblock, subblocks_per_seq, T, l_offset
        );
    }
}


// Backward kernel
__global__ void self_ref_backward_kernel(
    const float_accessor x,
    const float_accessor q,
    const float_accessor k,
    const float_accessor beta,
    const float_accessor y_diff,
    const float_accessor q_diff,
    const float_accessor k_diff,
    const float_accessor beta_diff,
    const float_accessor grad_out,
    float_accessor Wy,
    float_accessor Wq,
    float_accessor Wk,
    float_accessor wb,
    float_accessor grad_x,
    float_accessor grad_Wy,
    float_accessor grad_Wq,
    float_accessor grad_Wk,
    float_accessor grad_wb,
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
    float* shared_Wy = shared_mem;  // E*M
    float* shared_Wq = shared_mem + shared_kv_size;
    float* shared_Wk = shared_Wq + shared_kv_size;
    float* shared_wb = shared_Wk + shared_kv_size;

    float* shared_grad_Wy = shared_wb + 4*M;
    float* shared_grad_Wq = shared_grad_Wy + shared_kv_size;
    float* shared_grad_Wk = shared_grad_Wq + shared_kv_size;
    float* shared_grad_wb = shared_grad_Wk + shared_kv_size;

    float* shared_grad_q = shared_grad_wb + 4*M;
    float* shared_grad_k = shared_grad_q + M;
    float* shared_grad_beta = shared_grad_k + M;

    float* shared_grad_yq = shared_grad_beta + 4;
    float* shared_grad_qq = shared_grad_yq + M;
    float* shared_grad_kq = shared_grad_qq + M;
    float* shared_grad_bq = shared_grad_kq + M;

    float* grad_sft_cst = shared_grad_bq + 4;  // 1

    float* shared_grad_x = grad_sft_cst + 1;

    float* shared_gradout = shared_grad_x + M;
    float* shared_x = shared_gradout + M*T;
    float* shared_q = shared_x + E_block*T;
    float* shared_k = shared_q + M*T;
    float* shared_beta = shared_k + M*T;

    float* shared_ydiff = shared_beta + 4*T;
    float* shared_qdiff = shared_ydiff + M*T;
    float* shared_kdiff = shared_qdiff + M*T;
    float* shared_bdiff = shared_kdiff + M*T;

    if (threadIdx.x < M) {
        shared_grad_x[m] = 0.f;
        shared_grad_q[m] = 0.f;
        shared_grad_k[m] = 0.f;
        shared_grad_yq[m] = 0.f;
        shared_grad_qq[m] = 0.f;
        shared_grad_kq[m] = 0.f;
    }
    if (threadIdx.x < 4) {
        shared_grad_beta[m] = 0.f;
        shared_grad_bq[m] = 0.f;
    }
    if (threadIdx.x < 1) {
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
        shared_q[i] = q[n][h][t_bw][d];
        shared_k[i] = k[n][h][t_bw][d];

        shared_ydiff[i] = y_diff[n][h][t_bw][d];
        shared_qdiff[i] = q_diff[n][h][t_bw][d];
        shared_kdiff[i] = k_diff[n][h][t_bw][d];
    }
    for (int i = threadIdx.x; i < (t_end*E_block); i += blockDim.x)
    {
        int t = int(i / E_block) + l_offset;
        int t_bw = L - 1 - t;
        int d = (i % E_block);
        if (d < E) {
            shared_x[i] = x[n][h][t_bw][d];
        }
    }

    for (int i = threadIdx.x; i < (t_end*4); i += blockDim.x)
    {
        int t = int(i / 4) + l_offset;
        int t_bw = L - 1 - t;
        int d = i % 4;
        shared_beta[i] = beta[n][h][t_bw][d];
        shared_bdiff[i] = beta_diff[n][h][t_bw][d];
        // shared_bk[i] = beta_k[n][h][t_bw][d];
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
            shared_Wy[kv_idx] = Wy[n][h][e][m];
            shared_Wq[kv_idx] = Wq[n][h][e][m];
            shared_Wk[kv_idx] = Wk[n][h][e][m];

            shared_grad_Wy[kv_idx] = grad_Wy[n][h][e][m];
            shared_grad_Wq[kv_idx] = grad_Wq[n][h][e][m];
            shared_grad_Wk[kv_idx] = grad_Wk[n][h][e][m];

            if (m < 4) {
                shared_wb[e + m * E] = wb[n][h][e][m];
                shared_grad_wb[e + m * E] = grad_wb[n][h][e][m];
            }
        }
    }
    __syncthreads();

    // Main loop ==============================================================
    for (int t=0; t<t_end; t++) {
        int l = t + l_offset;
        int l_b = L - l -1;
        int m_abs = t*M + m;

        // ====================================================================
        // W_y part
        // ====================================================================
        // float y_diff = shared_yq[m_abs] - shared_yk[m_abs];
        float y_diff = shared_ydiff[m_abs];
        float y_ins = y_diff * shared_beta[t*4];  // beta[0] for y

        // revert Wy
        for (int sub=0; sub<subblocks_per_seq; sub++) {
            e = sub * E_per_subblock + e_local;
            e_abs = t*E_block + e;
            kv_idx = threadIdx.x + sub * blockDim.x;
            if (e < E) {
                shared_Wy[kv_idx] -= shared_k[e_abs] * y_ins;
            }
        }
        __syncthreads();

        // grad though update rule, level 1
        for (int sub=0; sub<subblocks_per_seq; sub++) {
            e = sub * E_per_subblock + e_local;
            e_abs = t*E_block + e;
            kv_idx = threadIdx.x + sub * blockDim.x;
            if (e < E) {
                // grad v
                float res = shared_k[e_abs] * shared_grad_Wy[kv_idx] 
                  * shared_beta[t*4];
                atomicAdd(
                    &shared_grad_yq[m],  // Define
                    res
                );  // grad_v done.

                // grad k part 1 and 2
                float res_k = shared_grad_Wy[kv_idx] * y_ins;
                atomicAdd(
                    &shared_grad_k[e],
                    res_k
                );

                // grad beta, with sigmoid
                float grad_b = shared_grad_Wy[kv_idx] * shared_k[e_abs]
                  * y_diff * shared_beta[t*4] * (1.f - shared_beta[t*4]);
                // beta[0] for y
                atomicAdd(
                    &shared_grad_beta[0],
                    grad_b
                );  // grad_beta done.
            }
        }
        __syncthreads();

        // level 2

        // from y_q to W_y, and from y_k to W_y
        for (int sub=0; sub<subblocks_per_seq; sub++) {
            e = sub * E_per_subblock + e_local;
            e_abs = t*E_block + e;
            kv_idx = threadIdx.x + sub * blockDim.x;
            if (e < E) {
                shared_grad_Wy[kv_idx] +=
                  shared_grad_yq[m] * (shared_q[e_abs] - shared_k[e_abs]);
            }
        }
        __syncthreads();

        // from y_q to q, and from y_k to k
        for (int sub=0; sub<subblocks_per_seq; sub++) {
            e = sub * E_per_subblock + e_local;
            e_abs = t*E_block + e;
            kv_idx = threadIdx.x + sub * blockDim.x;
            if (e < E) {
                float grad_q = shared_grad_yq[m] * shared_Wy[kv_idx];
                atomicAdd(
                    &shared_grad_q[e],
                    grad_q
                );  // from y_q to q
                float grad_k = - grad_q;
                atomicAdd(
                    &shared_grad_k[e],
                    grad_k
                );  // from y_k to k
            }
        }
        __syncthreads();

        // ====================================================================
        // W_q part
        // ====================================================================
        // float q_diff = shared_qq[m_abs] - shared_qk[m_abs];
        float q_diff = shared_qdiff[m_abs];
        float beta_q = shared_beta[t*4 + 1];  // beta[1] for q
        float q_ins = q_diff * beta_q;  // beta[1] for q

        // revert Wq
        for (int sub=0; sub<subblocks_per_seq; sub++) {
            e = sub * E_per_subblock + e_local;
            e_abs = t*E_block + e;
            kv_idx = threadIdx.x + sub * blockDim.x;
            if (e < E) {
                shared_Wq[kv_idx] -= shared_k[e_abs] * q_ins;
            }
        }
        __syncthreads();

        for (int sub=0; sub<subblocks_per_seq; sub++) {
            e = sub * E_per_subblock + e_local;
            e_abs = t*E_block + e;
            kv_idx = threadIdx.x + sub * blockDim.x;
            if (e < E) {
                // grad v
                float res = shared_k[e_abs] * shared_grad_Wq[kv_idx] * beta_q;
                atomicAdd(
                    &shared_grad_qq[m],  // Define
                    res
                );  // grad_v done.

                // grad k part 1 and 2
                float res_k = shared_grad_Wq[kv_idx] * q_ins;
                atomicAdd(
                    &shared_grad_k[e],
                    res_k
                );

                // grad beta, with sigmoid
                float grad_b = shared_grad_Wq[kv_idx] * shared_k[e_abs]
                  * q_diff * beta_q * (1.f - beta_q);
                // beta[0] for y
                atomicAdd(
                    &shared_grad_beta[1],  // 1 for q
                    grad_b
                );  // grad_beta done.
            }
        }
        __syncthreads();

        // from q_q to W_q, from q_k to W_q
        for (int sub=0; sub<subblocks_per_seq; sub++) {
            e = sub * E_per_subblock + e_local;
            e_abs = t*E_block + e;
            kv_idx = threadIdx.x + sub * blockDim.x;
            if (e < E) {
                shared_grad_Wq[kv_idx] +=
                  shared_grad_qq[m] * (shared_q[e_abs] - shared_k[e_abs]);
            }
        }
        __syncthreads();

        // from q_q to q, from q_q to k
        for (int sub=0; sub<subblocks_per_seq; sub++) {
            e = sub * E_per_subblock + e_local;
            e_abs = t*E_block + e;
            kv_idx = threadIdx.x + sub * blockDim.x;
            if (e < E) {
                float grad_q = shared_grad_qq[m] * shared_Wq[kv_idx];
                atomicAdd(
                    &shared_grad_q[e],
                    grad_q
                );
                float grad_k = - grad_q;
                atomicAdd(
                    &shared_grad_k[e],
                    grad_k
                );
            }
        }
        __syncthreads();

        // ====================================================================
        // w_b part
        // ====================================================================
        // float b_diff = shared_bq[m_abs] - shared_bk[m_abs];
        float beta_b = shared_beta[t*4 + 3];  // beta[3] for beta
        // float b_ins = b_diff * b_diff;

        // revert wb
        for (int sub=0; sub<subblocks_per_seq; sub++) {
            e = sub * E_per_subblock + e_local;
            e_abs = t*E_block + e;
            kv_idx = threadIdx.x + sub * blockDim.x;
            if (e < E && m < 4) {
                shared_wb[e + m * E] -= shared_k[e_abs] * beta_b
                  * shared_bdiff[t*4 + m];
                // shared_wb[kv_idx] -= shared_k[e_abs] * b_ins;
            }
        }
        __syncthreads();

        for (int sub=0; sub<subblocks_per_seq; sub++) {
            e = sub * E_per_subblock + e_local;
            e_abs = t*E_block + e;
            kv_idx = threadIdx.x + sub * blockDim.x;
            if (e < E  && m < 4) {
                // grad v
                float res = shared_k[e_abs] * shared_grad_wb[e + m * E] * beta_b;
                atomicAdd(
                    &shared_grad_bq[m],  // Define
                    res
                );  // grad_v done.

                // grad k part 1 and 2
                float res_k = shared_grad_wb[e + m * E] * beta_b
                  * shared_bdiff[t*4 + m];
                // float res_k = shared_grad_wb[kv_idx] * b_ins;
                atomicAdd(
                    &shared_grad_k[e],
                    res_k
                );

                // grad beta, with sigmoid
                float grad_b = shared_grad_wb[e + m * E] * shared_k[e_abs]
                  * shared_bdiff[t*4 + m] * beta_b * (1.f - beta_b);
                // beta[0] for y
                atomicAdd(
                    &shared_grad_beta[3],  // 3 for beta
                    grad_b
                );  // grad_beta done.
            }
        }
        __syncthreads();

        // from b_q to w_b, from b_k to w_b
        for (int sub=0; sub<subblocks_per_seq; sub++) {
            e = sub * E_per_subblock + e_local;
            e_abs = t*E_block + e;
            kv_idx = threadIdx.x + sub * blockDim.x;
            if (e < E && m < 4) {
                shared_grad_wb[e + m * E] +=
                  shared_grad_bq[m] * (shared_q[e_abs] - shared_k[e_abs]);
            }
        }
        __syncthreads();

        // from b_q to q, from b_q to k
        for (int sub=0; sub<subblocks_per_seq; sub++) {
            e = sub * E_per_subblock + e_local;
            e_abs = t*E_block + e;
            kv_idx = threadIdx.x + sub * blockDim.x;
            if (e < E && m < 4) {
                float grad_q = shared_grad_bq[m] * shared_wb[e + m * E];
                atomicAdd(
                    &shared_grad_q[e],
                    grad_q
                );
                float grad_k = - grad_q;
                atomicAdd(
                    &shared_grad_k[e],
                    grad_k
                );
            }
        }
        __syncthreads();

        // ====================================================================
        // W_k part
        // ====================================================================
        // float k_diff = shared_kq[m_abs] - shared_kk[m_abs];
        float k_diff = shared_kdiff[m_abs];
        float beta_k = shared_beta[t*4 + 2];  // beta[2] for k
        float k_ins = k_diff * beta_k;

        // revert Wk
        for (int sub=0; sub<subblocks_per_seq; sub++) {
            e = sub * E_per_subblock + e_local;
            e_abs = t*E_block + e;
            kv_idx = threadIdx.x + sub * blockDim.x;
            if (e < E) {
                shared_Wk[kv_idx] -= shared_k[e_abs] * k_ins;
            }
        }
        __syncthreads();

        for (int sub=0; sub<subblocks_per_seq; sub++) {
            e = sub * E_per_subblock + e_local;
            e_abs = t*E_block + e;
            kv_idx = threadIdx.x + sub * blockDim.x;
            if (e < E) {
                // grad v
                float res = shared_k[e_abs] * shared_grad_Wk[kv_idx] * beta_k;
                atomicAdd(
                    &shared_grad_kq[m],
                    res
                );  // grad_v done.

                // grad k part 1 and 2
                float res_k = shared_grad_Wk[kv_idx] * k_ins;
                atomicAdd(
                    &shared_grad_k[e],
                    res_k
                );

                // grad beta, with sigmoid
                float grad_b = shared_grad_Wk[kv_idx] * shared_k[e_abs]
                  * k_diff * beta_k * (1.f - beta_k);
                // beta[0] for y
                atomicAdd(
                    &shared_grad_beta[2],  // 1 for q
                    grad_b
                );  // grad_beta done.
            }
        }
        __syncthreads();

        // from k_q to W_k
        for (int sub=0; sub<subblocks_per_seq; sub++) {
            e = sub * E_per_subblock + e_local;
            e_abs = t*E_block + e;
            kv_idx = threadIdx.x + sub * blockDim.x;
            if (e < E) {
                shared_grad_Wk[kv_idx] +=
                  shared_grad_kq[m] * (shared_q[e_abs] - shared_k[e_abs]);
            }
        }
        __syncthreads();

        // from W_k to q
        for (int sub=0; sub<subblocks_per_seq; sub++) {
            e = sub * E_per_subblock + e_local;
            e_abs = t*E_block + e;
            kv_idx = threadIdx.x + sub * blockDim.x;
            if (e < E) {
                float grad_q = shared_grad_kq[m] * shared_Wk[kv_idx];
                atomicAdd(
                    &shared_grad_q[e],
                    grad_q
                );
                float grad_k = - grad_q;
                atomicAdd(
                    &shared_grad_k[e],
                    grad_k
                );
            }
        }
        __syncthreads();

        // ====================================================================
        // Grad though q
        // ====================================================================
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
        __syncthreads();  // end y to q
        if (threadIdx.x < 1) {
            grad_sft_cst[0] = 0.f;
        }
        __syncthreads();
        // from q to x.
        for (int sub=0; sub<subblocks_per_seq; sub++) {
            e = sub * E_per_subblock + e_local;
            e_abs = t*E_block + e;
            kv_idx = threadIdx.x + sub * blockDim.x;
            if (e < E) {
                float grad_x_q = shared_grad_q[m] * shared_Wq[kv_idx];
                atomicAdd(
                    &shared_grad_x[e],
                    grad_x_q
                );
            }
        }
        __syncthreads();
        // from q to W_q
        for (int sub=0; sub<subblocks_per_seq; sub++) {
            e = sub * E_per_subblock + e_local;
            e_abs = t*E_block + e;
            kv_idx = threadIdx.x + sub * blockDim.x;
            if (e < E) {
                shared_grad_Wq[kv_idx] += shared_grad_q[m] * shared_x[e_abs];
            }
        }
        __syncthreads();
        // ====================================================================
        // Grad though k
        // ====================================================================
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
        __syncthreads();  // end y to q
        if (threadIdx.x < 1) {
            grad_sft_cst[0] = 0.f;
        }
        __syncthreads();
        // from k to x.
        for (int sub=0; sub<subblocks_per_seq; sub++) {
            e = sub * E_per_subblock + e_local;
            e_abs = t*E_block + e;
            kv_idx = threadIdx.x + sub * blockDim.x;
            if (e < E) {
                float grad_x_k = shared_grad_k[m] * shared_Wk[kv_idx];
                atomicAdd(
                    &shared_grad_x[e],
                    grad_x_k
                );
            }
        }
        __syncthreads();
        // from k to W_k
        for (int sub=0; sub<subblocks_per_seq; sub++) {
            e = sub * E_per_subblock + e_local;
            e_abs = t*E_block + e;
            kv_idx = threadIdx.x + sub * blockDim.x;
            if (e < E) {
                shared_grad_Wk[kv_idx] += shared_grad_k[m] * shared_x[e_abs];
            }
        }
        __syncthreads();
        // ====================================================================
        // Grad though beta
        // ====================================================================
        // from beta to x
        for (int sub=0; sub<subblocks_per_seq; sub++) {
            e = sub * E_per_subblock + e_local;
            kv_idx = threadIdx.x + sub * blockDim.x;
            if (e < E && m < 4) {
                float grad_x_beta = shared_grad_beta[m] * shared_wb[e + m * E];
                atomicAdd(
                    &shared_grad_x[e],
                    grad_x_beta
                );
            }
        }
        __syncthreads();
        // from beta to w_b
        for (int sub=0; sub<subblocks_per_seq; sub++) {
            e = sub * E_per_subblock + e_local;
            e_abs = t*E_block + e;
            kv_idx = threadIdx.x + sub * blockDim.x;
            if (e < E && m < 4) {
                shared_grad_wb[e + m * E] +=
                  shared_grad_beta[m] * shared_x[e_abs];
            }
        }
        __syncthreads();
        // ====================================================================
        // Grad from current output
        // ====================================================================
        // grad: from out to x
        for (int sub=0; sub<subblocks_per_seq; sub++) {
            e = sub * E_per_subblock + e_local;
            kv_idx = threadIdx.x + sub * blockDim.x;
            if (e < E) {
                float grad_x_y = shared_gradout[m_abs] * shared_Wy[kv_idx];
                atomicAdd(
                    &shared_grad_x[e],
                    grad_x_y
                );
            }
        }
        __syncthreads();

        // grad: from out to Wy
        for (int sub=0; sub<subblocks_per_seq; sub++) {
            e = sub * E_per_subblock + e_local;
            e_abs = t*E_block + e;
            kv_idx = threadIdx.x + sub * blockDim.x;
            if (e < E) {
                shared_grad_Wy[kv_idx] +=
                  shared_gradout[m_abs] * shared_x[e_abs];
            }
        }
        __syncthreads();

        if (threadIdx.x < M) {
            atomicAdd(
                &grad_x[n][h][l_b][m],
                shared_grad_x[m]
            );
            shared_grad_x[m] = 0.f;
            shared_grad_q[m] = 0.f;
            shared_grad_k[m] = 0.f;
            shared_grad_yq[m] = 0.f;
            shared_grad_qq[m] = 0.f;
            shared_grad_kq[m] = 0.f;
        }
        if (threadIdx.x < 4) {
            shared_grad_beta[m] = 0.f;
            shared_grad_bq[m] = 0.f;
        }
        __syncthreads();
    }
    // grad for weights
    for (int sub=0; sub<subblocks_per_seq; sub++) {
        e = sub * E_per_subblock + e_local;
        kv_idx = threadIdx.x + sub * blockDim.x;
        if (e < E) {
            atomicAdd(
                &grad_Wy[n][h][e][m],
                shared_grad_Wy[kv_idx]
            );
            atomicAdd(
                &grad_Wq[n][h][e][m],
                shared_grad_Wq[kv_idx]
            );
            atomicAdd(
                &grad_Wk[n][h][e][m],
                shared_grad_Wk[kv_idx]
            );
            if (m < 4) {
                atomicAdd(
                    &grad_wb[n][h][e][m],
                    shared_grad_wb[e + m * E]
                );
            }
        }
        __syncthreads();
    }
}


// Backward
void self_ref_backward(
    const torch::Tensor x,
    const torch::Tensor q,
    const torch::Tensor k,
    const torch::Tensor beta,
    const torch::Tensor y_diff,
    const torch::Tensor q_diff,
    const torch::Tensor k_diff,
    const torch::Tensor beta_diff,
    const torch::Tensor grad_out,
    torch::Tensor Wy,
    torch::Tensor Wq,
    torch::Tensor Wk,
    torch::Tensor wb,
    torch::Tensor grad_x,
    torch::Tensor grad_Wy,
    torch::Tensor grad_Wq,
    torch::Tensor grad_Wk,
    torch::Tensor grad_wb
) {
//    const at::cuda::OptionalCUDAGuard device_guard(device_of(grad_queries));
    const int N = x.size(0);
    const int H = x.size(1);
    const int L = x.size(2);
    const int E = x.size(3);
    const int M = E;

    const int threads = 768;

    // First part ====================================
    int MPB = min(threads, E*M);
    // make sure that MUL_PER_BLOCK is divisible by M;
    MPB = int(MPB / M) *  M;
    const int subblocks_per_seq_value = ((E*M) + MPB - 1)/ MPB;
    const int E_per_subblock = MPB / M;
    const int blocks = N*H;
    const int E_block = E_per_subblock * subblocks_per_seq_value;

    // 2*E*M for KV and grad_KV, and 3*M.
    // int shared_mem_const = (5 * E_block + 5)*M + 2;
    // int shared_mem_const = (8 * E_block + 5 + 1)*M + 1;
    int shared_mem_const = (3 * E_block * M + 4 * M) * 2 + 2 * (3 * M + 4) + 1;

    // 5M for value, rnn_out, rnn_delayed, grad_out, tmp_grad.
    // E for key.
    int shared_mem_per_time = 2 * (3*M + 4) + E_block;

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
    cudaFuncSetAttribute(self_ref_backward_kernel,
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
    assert(T >= 1 &&
        "`d_head` too large. To obtain large models, keep `d_head` small"
        "e.g. 16 and increase the number of heads instead.");
    const int shared_mem =
      ((T*shared_mem_per_time) + shared_mem_const) * sizeof(float);

    for (int l_offset=0; l_offset < L; l_offset += T) {
        self_ref_backward_kernel
            <<<blocks, MPB, shared_mem>>>(
            x.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            q.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            k.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            beta.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            y_diff.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            q_diff.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            k_diff.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            beta_diff.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            grad_out.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            Wy.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            Wq.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            Wk.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            wb.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            grad_x.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            grad_Wy.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            grad_Wq.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            grad_Wk.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            grad_wb.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            N, H, L, E, M, E_per_subblock, subblocks_per_seq_value, T, l_offset
        );
    }
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "self_ref_forward",
        &self_ref_forward,
        "Compute the weighted sum of values but attending only to previous "
        "values."
    );
    m.def(
        "self_ref_backward",
        &self_ref_backward,
        "Compute the gradients for the fast weight memory."
    );
}

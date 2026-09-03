#pragma once

#include <cuda_pipeline.h>

#define WARP_SIZE 32
#define HALF_WARP_SIZE (WARP_SIZE / 2)

template<int QTileK, int CTileN, int HeadDim>
__device__ __forceinline__ void pipeline_prefetch_kv_tile(
    const float *__restrict__ gmem_src,
    float *__restrict__ shmem_dst,
    int lane_idx,
    int num_slices_per_warp)
{
    // The warp now prefetchs the entire chunk, without letting each half warp
    // load its slice.
    #pragma unroll
    for (int j = lane_idx;
             j < num_slices_per_warp * QTileK * CTileN / 4; j += WARP_SIZE) {

        // Coordinates within the tile
        const int tile_i = j / (2 * QTileK / 4);
        const int tile_j = j % (2 * QTileK / 4);

        const float *src = gmem_src + tile_i * HeadDim + 4 * tile_j;
        // Convert generic pointer to shared space state
        const uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(
            // TODO: Removing +4 boosts performance! Consider swizzling!
            shmem_dst + tile_i * (HeadDim + 4) + 4 * tile_j));

        asm volatile ("cp.async.cg.shared.global [%0], [%1], 16;"
                      : // No output registers
                      : "r"(dst), "l"(src));

        /*
        __pipeline_memcpy_async(
            shmem_dst + tile_i * (HeadDim + 4) + 4 * tile_j,
            gmem_src + tile_i * HeadDim + 4 * tile_j,
            sizeof(float4));
        */
    }

    // NOTE: ptx shows that an additional predicated cp.async is being
    // placed here (after the unrolled two), it's messing up ncu stats
}

// TODO: There's no reason to have HeadDim as template parameter,
// it should be a constexpr defined in a header which can be used
// by mla_api.cpp to check input sizes.
template<int QTileM, int QTileK, int CTileN, int HeadDim, int NumStages = 3>
__global__ void mla_decode_fused(const float *__restrict__ Q,
                                 const float *__restrict__ C,
                                 float *__restrict__ O,
                                 int N) // seq_length
{
    const int num_threads = QTileM * CTileN; // blockDim.x * blockDim.y;
    const int thread_idx = blockDim.x * threadIdx.y + threadIdx.x;
    const int num_warps = num_threads / WARP_SIZE;
    const int warp_idx = thread_idx / WARP_SIZE;
    const int lane_idx = thread_idx % WARP_SIZE;

    //const int HeadDimK = HeadDim;
    //const int HeadDimV = HeadDim - 64;

    extern __shared__ float shmem[] alignas(128);

    // TODO: Use swizzling instead of padding HeadDim

    float *__restrict__ Qs = shmem; // (QTileM, HeadDim)
    float *__restrict__ Cs = Qs + QTileM * HeadDim; // (CTileN, HeadDim + 4)
    float *__restrict__ Ps = Cs + CTileN * (HeadDim + 4); // (num_warps, QTileM, CTileN)

    for (int i = thread_idx; i < QTileM * HeadDim / 4; i += num_threads) {
        reinterpret_cast<float4 *>(Qs)[i] =
            reinterpret_cast<const float4 *>(Q)[
                (QTileM * blockIdx.y) * HeadDim / 4 + i];
    }

    __syncthreads();

    float prev_max = 0.0f; // max logit of prev Ps tile
    float prev_sum = 0.0f; // denominator of prev Ps tile

    const int TM_pc = 8;
    const int t_row = threadIdx.y / TM_pc;
    // NOTE: This is equivalent but increases runtime by 100us :/
    //const int t_row = warp_idx / num_warp_tiles_x;

    const int num_warp_tiles_y = QTileM / TM_pc;
    const int num_warp_tiles_x = num_warps / num_warp_tiles_y;

    float col_1[TM_pc] = { 0 };
    float col_2[TM_pc] = { 0 };
    float col_3[TM_pc] = { 0 };
    float col_4[TM_pc] = { 0 };

    // Tiled loop over attn matrix cols
    for (int C_j = blockIdx.x * N / gridDim.x;
             C_j < (blockIdx.x + 1) * N / gridDim.x; C_j += CTileN) {
        // Load Cs once, reuse it for the projection
        // Should I worry about transposing Cs?

        // Each warp prefetches a (CTileN, QTileK) tile from C
        // for each stage of the pipeline

        // Thread tile shape (Tm, Tn) for PC^T product
        const int TN = 2;
        const int TM = 8;

        const int num_t_cols = CTileN / TN;
        const int num_t_rows = QTileM / TM;

        const int num_t_per_slice = num_t_cols * num_t_rows;
        const int num_slices_per_warp = WARP_SIZE / num_t_per_slice;

        // thread tile coordinates
        // WARNING: Harcoded for QK=CN=16 TN=2 TM=8
        const int tx = (lane_idx % num_t_per_slice) % QTileK;
        const int ty = lane_idx / 16;

        //const int tx = (lane_idx % num_t_per_slice) % num_t_cols;
        //const int ty = (lane_idx % num_t_per_slice) / num_t_cols;

        const int stage_width = num_slices_per_warp * QTileK * num_warps;
        const int warp_c_offset = num_slices_per_warp * QTileK * warp_idx;// +
                                  //QTileK * (lane_idx / num_t_per_slice);

        const int slice_offset =
            QTileK * ((lane_idx % num_t_per_slice) / QTileK);

        #pragma unroll
        for (int s = 0; s < NumStages - 1; ++s) {
            // Note that before wrapping the prefetch in a function,
            // the kernel was slightly faster and used more registers
            // TODO: Avoid excessive wavefronts or global requests
            // (perhaps padding is the culprit?) Look at the ptx
            pipeline_prefetch_kv_tile<QTileK, CTileN, HeadDim>(
                C + C_j * HeadDim + warp_c_offset + s * stage_width,
                Cs + warp_c_offset + s * stage_width,
                lane_idx, num_slices_per_warp);
            __pipeline_commit();
        }

        float logs[TM][TN] = { 0 };

        // Each warp accumulates a partial product (slice-k),
        // while each thread computes a column of QTileM values
        // by accumulating outer products
        for (int Q_j = warp_c_offset;
                 Q_j < HeadDim; Q_j += stage_width) {

            // Prefetchng the next tiles
            if (Q_j < HeadDim - (NumStages - 1) * stage_width) {
                // TODO: Warp-specialization for consumer-producer pattern?
                pipeline_prefetch_kv_tile<QTileK, CTileN, HeadDim>(
                    C + C_j * HeadDim + Q_j + (NumStages - 1) * stage_width,
                    Cs + Q_j + (NumStages - 1) * stage_width,
                    lane_idx, num_slices_per_warp);
            }

            // Commit even when pipeline is empty for better codegen
            __pipeline_commit(); // cp.async.commit_group;

            // TODO: Wait for partial tile loads instead of the full tile
            __pipeline_wait_prior(NumStages - 1); // cp.async.wait_group N;
            __syncwarp();

            float Qr[TM][QTileK];

            for (int k = 0; k < QTileK; ++k) {
                #pragma unroll
                for (int m = 0; m < TM; ++m) {
                    Qr[m][k] = Qs[(ty * TM + m) * HeadDim +
                                  Q_j + slice_offset + k];
                }
            }

            for (int k = 0; k < QTileK; ++k) {
                #pragma unroll
                for (int n = 0; n < TN; ++n) {
                    float Cr = Cs[(n * num_t_cols + tx) *
                                  (HeadDim + 4) + Q_j + slice_offset + k];
                    #pragma unroll
                    for (int m = 0; m < TM; ++m) {
                        logs[m][n] += Cr * Qr[m][k];
                    }
                }
            }
        }

        #pragma unroll
        for (int m = 0; m < TM; ++m) {
            #pragma unroll
            for (int n = 0; n < TN; ++n) {
                // Reduce wrt to the other warp slices
                /*
                #pragma unroll
                for (int s = num_slices_per_warp; s > 1; s >>= 1) {
                    logs[m][n] +=
                        __shfl_xor_sync(0xffffffff, logs[m][n], WARP_SIZE / s);
                }
                */
                // WARNING: Works for QM=CN=8 TN=2 TM=8
                logs[m][n] +=
                    __shfl_xor_sync(0xffffffff, logs[m][n], 8);
            }
        }

        // NOTE: no need to guard, all threads have the accumulated values
        //if (lane_idx < num_t_per_slice) {
            #pragma unroll
            for (int m = 0; m < TM; ++m) {
                const int warp_offset = QTileM * CTileN * warp_idx;

                // WARNING: Can't index logs[m][..] using a runtime variable..

                // TODO: Not great, this loses some performance. Consider
                // hardcoding individual col arrays if TN=2 performs best.
                #pragma unroll
                for (int n = 0; n < TN; ++n) {
                    Ps[warp_offset + (ty * TM + m) * CTileN
                                   + tx + n * num_t_cols] = logs[m][n];
                }
            }
        //}

        __syncthreads();

        // Reduce logs across warps, one element per thread
        // TODO: Tree reduction?
        float log = 0;
        for (int w = 0; w < num_warps; ++w) {
            log += Ps[QTileM * CTileN * w + threadIdx.y * CTileN + threadIdx.x];
        }

        log /= sqrtf(HeadDim);

        // TODO: The softmax section could probably be revisited.

        // As long as CTileN <= 32, use warp shuffles to compute partial reduction

        float max = log;
        // Butterfly warp reduction
        for (int s = CTileN >> 1; s > 0; s >>= 1) {
            max = fmaxf(max, __shfl_xor_sync(0xffffffff, max, s));
        }

        max = fmaxf(prev_max, max);

        float exp_log = expf(log - max);
        float sum = exp_log;
        for (int s = CTileN >> 1; s > 0; s >>= 1) {
            sum += __shfl_xor_sync(0xffffffff, sum, s);
        }

        // Online softmax correction
        float norm_coef = expf(prev_max - max);
        sum += prev_sum * norm_coef;

        prev_max = max;
        prev_sum = sum;

        __shared__ float norm_coefs[QTileM];
        // Can I avoid this? I guess not
        if (threadIdx.x == 0) {
            norm_coefs[threadIdx.y] = norm_coef;
        }

        // Load exp logit to shmem
        Ps[threadIdx.y * blockDim.x + threadIdx.x] = exp_log;

        __syncthreads();

        // WARNING: For the PC product I am supposing thread tiles span
        // the entire warp tile vertically! For QTileM=CTileN=16 each
        // thread works only on the same 4 cols.

        //for (int j = 4 * (warp_idx % num_warp_tiles_x) * WARP_SIZE;
        //         j < HeadDim - 64; j += 4 * WARP_SIZE * num_warp_tiles_x) {

        for (int m = 0; m < TM_pc; ++m) {
            float norm_coef = norm_coefs[t_row * TM_pc + m];
            // Online softmax normalization
            col_1[m] *= norm_coef;
            col_2[m] *= norm_coef;
            col_3[m] *= norm_coef;
            col_4[m] *= norm_coef;
        }

        // NOTE: Ps loads were not vectorized anymore,
        // but swapping the loop order is enough to bring them back!
        // (otherwise transposing Ps would work as well).
        #pragma unroll
        for (int m = 0; m < TM_pc; ++m) {

            for (int k = 0; k < CTileN; ++k) {
                // Compiler does this already automatically here..
                float4 C_val = reinterpret_cast<float4 *>(Cs)[
                    k * (HeadDim + 4) / 4 +
                        (warp_idx % num_warp_tiles_x) * WARP_SIZE + lane_idx];

                    // TODO: Consider transposing Ps, it seems beneficial
                    // even if we are already able to issue 128 bit loads.
                    //float P_val = Ps[k * QTileM + t_row * TM_pc + m];
                    float P_val = Ps[(t_row * TM_pc + m) * CTileN + k];


                    col_1[m] += P_val * C_val.x;
                    col_2[m] += P_val * C_val.y;
                    col_3[m] += P_val * C_val.z;
                    col_4[m] += P_val * C_val.w;
            }
        }

        __syncthreads(); // Wait before reusing Cs
    }

    for (int m = 0; m < TM_pc; ++m) {
        float4 O_new = { col_1[m], col_2[m], col_3[m], col_4[m] };

        reinterpret_cast<float4 *>(O)[
            (blockIdx.y * QTileM + t_row * TM_pc + m) * (HeadDim - 64) / 4 +
                (warp_idx % num_warp_tiles_x) * WARP_SIZE + lane_idx] = O_new;
    }

    // TODO: I could avoid this and shfl the prev_sum values,
    // writing only once to gmem. Caches are probably hiding the
    // following additional transfer.
    __syncthreads();

    // It looks like fully contiguous writes within the warp
    // are still better, even if sectors are not split across two requests..
    // I could use shfl to share the prev_sum values.

    for (int j = 0; j < (HeadDim - 64) / 4; j += CTileN) {
        float4 O_val = reinterpret_cast<float4 *>(O)[
            (blockIdx.y * QTileM + threadIdx.y) * (HeadDim - 64) / 4 +
                j + threadIdx.x];

        float4 O_val_norm = { __fdividef(O_val.x, prev_sum),
                              __fdividef(O_val.y, prev_sum),
                              __fdividef(O_val.z, prev_sum),
                              __fdividef(O_val.w, prev_sum) };

        reinterpret_cast<float4 *>(O)[
            (blockIdx.y * QTileM + threadIdx.y) * (HeadDim - 64) / 4 +
                j + threadIdx.x] = O_val_norm;
    }
}

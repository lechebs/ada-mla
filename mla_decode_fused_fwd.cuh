#pragma once

#include <cuda_pipeline.h>

#define WARP_SIZE 32

template<int QTileK, int CTileN, int HeadDim>
__device__ __forceinline__ void pipeline_prefetch_kv_tile(
    const float *__restrict__ gmem_src,
    float *__restrict__ shmem_dst)
{
    #pragma unroll
    for (int j = threadIdx.x; j < QTileK * CTileN / 4; j += WARP_SIZE) {
        // Coordinates within the tile
        const int tile_i = j / (QTileK / 4);
        const int tile_j = j % (QTileK / 4);
        __pipeline_memcpy_async(
            shmem_dst + tile_i * (HeadDim + 4) + 4 * tile_j,
            gmem_src + tile_i * HeadDim + 4 * tile_j,
            sizeof(float4));
    }
}

// Template args not strictly needed anymore since shmem is dynamic
// but they're probably going to be useful to unroll some loops
template<int QTileM, int QTileK, int CTileN, int HeadDim, int NumStages = 3>
__global__ void mla_decode_fused(const float *__restrict__ Q,
                                 const float *__restrict__ C,
                                 float *__restrict__ O,
                                 int M, // num_heads
                                 int N) // seq_length
{
    extern __shared__ float shmem[];

    float *Qs = shmem;
    float *Cs = Qs + QTileM * HeadDim;
    // TODO: Padding Cs causes excessive shared accesses
    float *Ps = Cs + (HeadDim + 4) * CTileN;

    const int num_warps = QTileM * CTileN / WARP_SIZE;
    const int warp_id = threadIdx.y;

    // Ps has shape (num_warps, QTileM, CTileN)

    for (int i = blockDim.x * threadIdx.y + threadIdx.x;
             i < QTileM * HeadDim; i += QTileM * CTileN) {
        Qs[i] = Q[(QTileM * blockIdx.y) * HeadDim + i];
    }

    __syncthreads();

    float prev_max = 0.0f; // max logit of prev tile
    float prev_sum = 0.0f; // denominator of prev tile

    // Tiled loop over attn matrix cols
    for (int C_j = 0; C_j < N; C_j += CTileN) {

        // Load Cs once, reuse it for the projection
        // Should I worry about transposing Cs?

        // Each warp prefetches a (CTileN, QTileK) tile from C
        // for each stage of the pipeline
        const int stage_width = QTileK * num_warps;
        const int warp_c_offset = QTileK * warp_id;

        #pragma unroll
        for (int s = 0; s < NumStages - 1; ++s) {
            // Note that before wrapping the prefetch in a function,
            // the kernel was slightly faster and used more registers
            pipeline_prefetch_kv_tile<QTileK, CTileN, HeadDim>(
                C + C_j * HeadDim + warp_c_offset + s * stage_width,
                Cs + warp_c_offset + s * stage_width);
            __pipeline_commit();
        }

        float logs[QTileM] = { 0 }; // logits

        // Each warp accumulates a partial product (slice-k),
        // while each thread computes a column of QTileM values
        // by accumulating outer products
        for (int Q_j = warp_c_offset;
                 Q_j < HeadDim; Q_j += QTileK * num_warps) {

            // Prefetching the next tiles
            if (Q_j < HeadDim - (NumStages - 1) * stage_width) {
                // TODO: Warp-specialization for consumer-producer pattern?
                pipeline_prefetch_kv_tile<QTileK, CTileN, HeadDim>(
                    C + C_j * HeadDim + Q_j + (NumStages - 1) * stage_width,
                    Cs + Q_j + (NumStages - 1) * stage_width);
            }
            // Commit even when pipeline is empty for better codegen
            __pipeline_commit();

            __pipeline_wait_prior(NumStages - 1);
            __syncwarp();

            for (int k = 0; k < QTileK; ++k) {
                // Tile row into registers
                float C_val = Cs[threadIdx.x * (HeadDim + 4) + Q_j + k];

                // Unroll so that logs gets promoted to registers
                #pragma unroll
                for (int m = 0; m < QTileM; ++m) {
                    // TODO: Avoid bank conflicts
                    logs[m] += C_val * Qs[m * HeadDim + Q_j + k];
                }
            }

        }

        #pragma unroll
        for (int m = 0; m < QTileM; ++m) {
            Ps[QTileM * CTileN * warp_id + m * CTileN + threadIdx.x] =
                logs[m];
        }

        __syncthreads();

        // Reduce logs across warps, one element per thread
        // TODO: Tree reduction?
        float log = 0;
        for (int w = 0; w < num_warps; ++w) {
            log += Ps[QTileM * CTileN * w + threadIdx.y * CTileN + threadIdx.x];
        }

        log /= sqrtf(HeadDim);

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
        if (threadIdx.x / warpSize == 0) {
            norm_coefs[warp_id] = norm_coef;
        }

        // Load exp logit to shmem
        Ps[threadIdx.y * blockDim.x + threadIdx.x] = exp_log;

        __syncthreads();

        for (int j = 2 * warp_id * CTileN;
                 j < HeadDim; j += 2 * CTileN * num_warps) {

            // Each thread computes two adjacent
            // columns of the resulting tile
            float col_1[QTileM];
            float col_2[QTileM];

            // Online softmax correction
            #pragma unroll
            for (int m = 0; m < QTileM; ++m) {
                float norm_coef = norm_coefs[m];
                // Force 8 byte loads to allow coalescing
                float2 O_curr = reinterpret_cast<float2 *>(O)[
                    (blockIdx.y * QTileM + m) * HeadDim / 2
                        + j / 2 + threadIdx.x];

                col_1[m] = O_curr.x * norm_coef;
                col_2[m] = O_curr.y * norm_coef;
            }

            // TODO: Try letting each warp compute multiple tiles
            // so that Qs values can be reused!
            for (int k = 0; k < CTileN; ++k) {
                // Compiler does this already automatically here..
                float2 C_val = reinterpret_cast<float2 *>(Cs)[
                    k * (HeadDim + 4) / 2 + j / 2 + threadIdx.x];

                #pragma unroll
                for (int m = 0; m < QTileM; ++m) {
                    float P_val = Ps[m * CTileN + k];
                    col_1[m] += P_val * C_val.x;
                    col_2[m] += P_val * C_val.y;
                }
            }

            #pragma unroll
            for (int m = 0; m < QTileM; ++m) {
                float2 O_new = { col_1[m], col_2[m] };
                reinterpret_cast<float2 *>(O)[
                    (blockIdx.y * QTileM + m) * HeadDim / 2
                        + j / 2 + threadIdx.x] = O_new;
            }
        }

        __syncthreads(); // Wait before reusing Cs
    }

    for (int j = 0; j < HeadDim; j += CTileN) {
        O[(blockIdx.y * QTileM + threadIdx.y) * HeadDim +
            j + threadIdx.x] /= prev_sum;
    }
}

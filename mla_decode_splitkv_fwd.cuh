#pragma once

// TODO: Find a way to get around this

#ifdef __CUDA_NO_HALF_OPERATORS__
#undef __CUDA_NO_HALF_OPERATORS__
#endif

#ifdef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#endif

#include <cuda_fp16.h>

template<int QTileM, int QTileK, int CTileN, int HeadDim, typename Scalar>
__global__ void mla_decode_splitkv(const Scalar *__restrict__ Q,
                                   const Scalar *__restrict__ C,
                                   Scalar *__restrict__ O,
                                   Scalar *__restrict__ max,
                                   Scalar *__restrict__ sum,
                                   int M, // num_heads
                                   int N, // seq_length
                                   int num_splits)
{
    // TODO: Probably better to specialize struct that returns
    // a pointer to extern __shared__
    extern __shared__ __align__(sizeof(Scalar)) unsigned char shmem[];

    Scalar *Qs = reinterpret_cast<Scalar *>(shmem);
    Scalar *Cs = Qs + QTileM * HeadDim;
    Scalar *Ps = Cs + HeadDim * CTileN;

    // WARNING: out is initialized with at::empty,
    // while here I'm supposing is initialized with at::zeros

    for (int i = blockDim.x * threadIdx.y + threadIdx.x;
             i < QTileM * HeadDim;
             i += QTileM * CTileN) {
        Qs[i] = Q[(QTileM * blockIdx.y) * HeadDim + i];
    }

    __syncthreads();

    Scalar prev_max = 0.0f; // max logit of prev tile
    Scalar prev_sum = 0.0f; // denominator of prev tile

    int split_N = N / num_splits;
    // Tiled loop over attn matrix cols
    for (int C_j = blockIdx.x * split_N;
             C_j < (blockIdx.x + 1) * split_N;
             C_j += CTileN) {

        // Load Cs once, reuse it for the projection
       // Should I worry about transposing Cs? I guess not

        // Unrolling here increases performance,
        // compiler does it automatically
        // #pragma unroll 8
        for (int j = threadIdx.y * blockDim.x + threadIdx.x;
                 j < CTileN * HeadDim;
                 j += QTileM * CTileN) {
            Cs[j] = C[C_j * HeadDim + j];
        }

        __syncthreads();

        Scalar log = 0.0f; // Logit
        // Tiled loop over query cols
        // TODO: At this point QTileK is not used, but it will
        // be used to better parallelize across warps
        for (int Q_j = 0; Q_j < HeadDim; Q_j += QTileK) {

            for (int k = 0; k < QTileK; ++k) {
                log += Qs[threadIdx.y * HeadDim + Q_j + k] *
                       // Cs is transposed
                       Cs[threadIdx.x * HeadDim + Q_j + k];
            }
        }

        log /= sqrtf(HeadDim);

        // As long as CTileN <= 32, use warp shuffles to compute partial reduction

        Scalar max = log;
        // Butterfly reduction within quarter warp
        #pragma unroll
        for (int s = CTileN >> 1; s > 0; s >>= 1) {
            max = fmaxf(max, __shfl_xor_sync(0xffffffff, max, s));
        }

        max = fmaxf(prev_max, max);

        Scalar exp_log = expf(log - max);
        Scalar sum = exp_log;
        #pragma unroll
        for (int s = CTileN >> 1; s > 0; s >>= 1) {
            sum += __shfl_xor_sync(0xffffffff, sum, s);
        }

        // Online softmax correction
        Scalar norm_coef = expf(prev_max - max);
        sum += prev_sum * norm_coef;

        prev_max = max;
        prev_sum = sum;

        // Load exp logit to shmem
        Ps[threadIdx.y * blockDim.x + threadIdx.x] = exp_log;

        __syncthreads();

        for (int j = 0; j < HeadDim; j += CTileN) {

            // Online softmax correction
            O[blockIdx.x * HeadDim * M
                + (blockIdx.y * QTileM + threadIdx.y) * HeadDim
                + j + threadIdx.x] *= norm_coef;
            //Os[O_i * HeadDim + j + O_j] *= norm_coef;

            for (int k = 0; k < CTileN; ++k) {
                O[blockIdx.x * HeadDim * M
                    + (blockIdx.y * QTileM + threadIdx.y) * HeadDim
                    + j + threadIdx.x] +=
                //Os[O_i * HeadDim + j + O_j] +=
                    Ps[threadIdx.y * CTileN + k] *
                    Cs[k * HeadDim + j + threadIdx.x];
            }
        }

        __syncthreads(); // Wait before reusing Cs
    }

    // Normalize in the combine kernel
    /*
    for (int j = 0; j < HeadDim; j += CTileN) {
        O[(blockIdx.y * QTileM + threadIdx.y) * HeadDim +
            j + threadIdx.x] /= prev_sum;
        //O[(blockIdx.y * QTileM + O_i) * HeadDim + j + O_j] =
        //    Os[O_i * HeadDim + j + O_j] / prev_sum;
    }
    */

    // Save prev_max and prev_sum for each row, worry about coalescing later
    if (threadIdx.x == 0) { // not strictly needed, all threads have the same value
        max[M * blockIdx.x + QTileM * blockIdx.y + threadIdx.y] = prev_max;
        sum[M * blockIdx.x + QTileM * blockIdx.y + threadIdx.y] = prev_sum;
    }

}

__global__ void mla_decode_combine()
{

}

template<typename T>
void run_mla_decode_splitkv_(const T *query,
                             const T *cache,
                             T *out,
                             T *splits_max,
                             T *splits_sum,
                             T *splits_out,
                             int num_splits,
                             int num_heads,
                             int head_dim_c,
                             int seq_length)
{
    const int q_tile_m = 8;
    const int q_tile_k = 16;
    const int c_tile_n = 32;

    dim3 block(c_tile_n, q_tile_m);
    dim3 grid(num_splits, (num_heads - 1) / block.y + 1);

    // TODO: Compute size using input args
    int shmem_bytes = (q_tile_m * head_dim_c +
                       c_tile_n * head_dim_c +
                       q_tile_m * c_tile_n) * sizeof(T);

    // Dynamic shmem is required for allocations > 48KB
    cudaError_t error = cudaFuncSetAttribute(
        mla_decode_splitkv<q_tile_m, q_tile_k, c_tile_n, 576, T>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shmem_bytes);

    if (error != cudaSuccess) {
        printf("Error: %s (%s)\n",
               cudaGetErrorName(error),
               cudaGetErrorString(error));
    }

    mla_decode_splitkv<q_tile_m, q_tile_k, c_tile_n, 576, T>
        <<<grid, block, shmem_bytes>>>(
            query, cache, splits_out, splits_max, splits_sum,
            num_heads, seq_length, num_splits);

    //mla_decode_combine<>

    if ((error = cudaGetLastError()) != cudaSuccess) {
        printf("Error: %s (%s)\n",
               cudaGetErrorName(error),
               cudaGetErrorString(error));
    }
}

#pragma once

// Template args not strictly needed anymore since shmem is dynamic
// but they're probably going to be useful to unroll some loops
template<int QTileM, int QTileK, int CTileN, int HeadDim>
__global__ void mla_decode_fused(const float *__restrict__ Q,
                                 const float *__restrict__ C,
                                 float *__restrict__ O,
                                 int M, // num_heads
                                 int N) // seq_length
{
    extern __shared__ float shmem[];

    float *Qs = shmem;
    float *Cs = Qs + QTileM * HeadDim;
    float *Ps = Cs + HeadDim * CTileN;

    /*
    for (int j = threadIdx.y * blockDim.x + threadIdx.x;
             j < QTileM * HeadDim;
             j += blockDim.x * blockDim.y) {
        Os[j] = 0.0f;
    }
    */

    // WARNING: out is initialized with at::empty,
    // while here I'm supposing is initialized with at::zeros

    for (int i = blockDim.x * threadIdx.y + threadIdx.x;
             i < QTileM * HeadDim;
             i += QTileM * CTileN) {
        Qs[i] = Q[(QTileM * blockIdx.y) * HeadDim + i];
    }

    __syncthreads();

    float prev_max = 0.0f; // max logit of prev tile
    float prev_sum = 0.0f; // denominator of prev tile

    // Tiled loop over attn matrix cols
    for (int C_j = 0; C_j < N; C_j += CTileN) {

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

        float log = 0.0f; // Logit
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

        float max = log;
        // Butterfly reduction within quarter warp
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

        // Load exp logit to shmem
        Ps[threadIdx.y * blockDim.x + threadIdx.x] = exp_log;

        __syncthreads();

        for (int j = 0; j < HeadDim; j += CTileN) {

            // Online softmax correction
            O[(blockIdx.y * QTileM + threadIdx.y) * HeadDim +
                j + threadIdx.x] *= norm_coef;
            //Os[O_i * HeadDim + j + O_j] *= norm_coef;

            for (int k = 0; k < CTileN; ++k) {
                O[(blockIdx.y * QTileM + threadIdx.y) * HeadDim +
                    j + threadIdx.x] +=
                //Os[O_i * HeadDim + j + O_j] +=
                    Ps[threadIdx.y * CTileN + k] *
                    Cs[k * HeadDim + j + threadIdx.x];
            }
        }

        __syncthreads(); // Wait before reusing Cs
    }

    for (int j = 0; j < HeadDim; j += CTileN) {
        O[(blockIdx.y * QTileM + threadIdx.y) * HeadDim +
            j + threadIdx.x] /= prev_sum;
        //O[(blockIdx.y * QTileM + O_i) * HeadDim + j + O_j] =
        //    Os[O_i * HeadDim + j + O_j] / prev_sum;
    }
}

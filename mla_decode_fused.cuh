#pragma once

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
    //float *Os = Cs + HeadDim * CTileN;

    /*
    __shared__ float Qs[QTileM * QTileK];
    __shared__ float Cs[HeadDim * CTileN];
    __shared__ float Os[QTileM * HeadDim];
    */

    /*
    for (int j = threadIdx.y * blockDim.x + threadIdx.x;
             j < QTileM * HeadDim;
             j += blockDim.x * blockDim.y) {
        Os[j] = 0.0f;
    }
    */

    for (int i = blockDim.x * threadIdx.y + threadIdx.x;
             i < QTileM * HeadDim;
             i += blockDim.x * blockDim.y) {
        Qs[i] = Q[(QTileM * blockIdx.y) * HeadDim + i];
    }

    __syncthreads();

    float prev_max = 0.0f; // max logit of prev tile
    float prev_sum = 0.0f; // denominator of prev tile

    // Tiled loop over attn matrix cols
    for (int C_j = 0; C_j < N; C_j += CTileN) {

        // Load Cs once, reuse it for the projection
        // I have 16x16 blocks, so I need to load
        // in chunks, note that C is transposed in memory
        // Should I worry about transposing Cs? I guess not

        // The block size could be passed as template param
        for (int j = threadIdx.y * blockDim.x + threadIdx.x;
                 j < CTileN * HeadDim;
                 j += blockDim.x * blockDim.y) {
            // NOTE: Cs could be loaded in chunks of size (CTileN, QTileM),
            // big enough to perform the matmul with the corresponding Qs
            // tiles. However, make sure that the accesses remaing coalesced
            // and that vector loads can be used.
            Cs[j] = C[C_j * HeadDim + j];
        }

        __syncthreads();

        // Qs tiles are (QTileM, QTileK) = (32, 16)
        // Cs tiles are (CTileN, HeadDim) = (8, 576)

        float log = 0.0f; // Logit
        // Tiled loop over query cols
        for (int Q_j = 0; Q_j < HeadDim; Q_j += QTileK) {

            // Enough threads in a block to assign one output
            // to each one. From (16, 16) go to (32, 8) by assigning
            // half of the threads in the block the upper half of As
            const int A_i = threadIdx.y * 2 + (threadIdx.x >= blockDim.x / 2);
            const int A_j = threadIdx.x % CTileN;

            for (int k = 0; k < QTileK; ++k) {
                log += Qs[A_i * HeadDim + Q_j + k] *
                       // Cs is transposed
                       Cs[A_j * HeadDim + Q_j + k];
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

        // TODO: Make blocks (32, 8)
        // Perhaps better to name these "O_tile_i" while
        // "Q_tile_j" is actually "Q_j"
        const int O_i = threadIdx.y * 2 + (threadIdx.x >= blockDim.x / 2);
        const int O_j = threadIdx.x % CTileN;

        for (int j = 0; j < HeadDim; j += CTileN) {

            // Online softmax correction
            O[(blockIdx.y * QTileM + O_i) * HeadDim + j + O_j] *= norm_coef;
            //Os[O_i * HeadDim + j + O_j] *= norm_coef;

            for (int k = 0; k < CTileN; ++k) {
                O[(blockIdx.y * QTileM + O_i) * HeadDim + j + O_j] +=
                //Os[O_i * HeadDim + j + O_j] +=
                    Ps[O_i * CTileN + k] * Cs[k * HeadDim + j + O_j];
            }
        }

        __syncthreads(); // Wait before reusing Cs
    }

    const int O_i = threadIdx.y * 2 + (threadIdx.x >= QTileK / 2);
    const int O_j = threadIdx.x % CTileN;

    for (int j = 0; j < HeadDim; j += CTileN) {
        O[(blockIdx.y * QTileM + O_i) * HeadDim + j + O_j] /= prev_sum;
        //O[(blockIdx.y * QTileM + O_i) * HeadDim + j + O_j] =
        //    Os[O_i * HeadDim + j + O_j] / prev_sum;
    }
}

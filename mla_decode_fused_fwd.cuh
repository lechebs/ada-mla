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
    // TODO: Qs could be reused to hold Ps values
    float *Ps = Cs + HeadDim * CTileN;

    const int num_warps = QTileM * CTileN / 32;
    const int warp_id = threadIdx.y;

    // Ps has shape (num_warps, QTileM, CTileN)

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

        #pragma unroll 4
        for (int j = threadIdx.y * blockDim.x + threadIdx.x;
                 j < CTileN * HeadDim / 4;
                 j += QTileM * CTileN) {
            // TODO: Load chunks asynchronously while
            // performing products to pipeline?
            // Warp-specialization for consumer-producer pattern?
            // I can afford blocks with more threads..
            reinterpret_cast<float4 *>(Cs)[j] =
                reinterpret_cast<const float4 *>(C)[C_j * HeadDim / 4 + j];
        }

        __syncthreads();

        float logs[QTileM] = { 0 };

        // Each warp accumulates a partial product (slice-k),
        // while each thread computes a column of QTileM values
        // by accumulating outer products
        for (int Q_j = QTileK * warp_id;
                 Q_j < HeadDim; Q_j += QTileK * num_warps) {

            for (int k = 0; k < QTileK; ++k) {
                // Tile row into registers
                // TODO: Avoid bank conflicts
                float C_val = Cs[threadIdx.x * HeadDim + Q_j + k];

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
            // TODO: Qs could be reused to perform reduction
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

        __shared__ float norm_coefs[QTileM];
        // Can I avoid this? I guess not
        if (threadIdx.x / warpSize == 0) {
            norm_coefs[warp_id] = norm_coef;
        }

        // Load exp logit to shmem
        Ps[threadIdx.y * blockDim.x + threadIdx.x] = exp_log;

        __syncthreads();

        for (int j = warp_id * CTileN; j < HeadDim; j += CTileN * num_warps) {

            // Each thread computes a column of results
            float col[QTileM];

            // Online softmax correction
            #pragma unroll
            for (int m = 0; m < QTileM; ++m) {
                col[m] = O[(blockIdx.y * QTileM + m) * HeadDim
                    + j + threadIdx.x] * norm_coefs[m];
            }

            // TODO: Try letting each warp compute multiple tiles
            // so that Qs values can be reused!
            for (int k = 0; k < CTileN; ++k) {
                float C_val = Cs[k * HeadDim + j + threadIdx.x];

                #pragma unroll
                for (int m = 0; m < QTileM; ++m) {
                    col[m] += Ps[m * CTileN + k] * C_val;
                }
            }

            #pragma unroll
            for (int m = 0; m < QTileM; ++m) {
                O[(blockIdx.y * QTileM + m) * HeadDim + j + threadIdx.x] =
                    col[m];
            }
        }

        __syncthreads(); // Wait before reusing Cs
    }

    for (int j = 0; j < HeadDim; j += CTileN) {
        O[(blockIdx.y * QTileM + threadIdx.y) * HeadDim +
            j + threadIdx.x] /= prev_sum;
    }
}

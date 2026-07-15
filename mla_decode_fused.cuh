#pragma once

template<int Tm, int Tn, int Tk>
__global__ void mla_decode_fused(const float *Q,
                                 const float *C,
                                 float *O,
                                 int m, // num_heads
                                 int n, // seq_length
                                 int k) // head_dim
{
    // Keep the same tiling strategy as the naive kernels
    // so 16x16 thread blocks

    // Coordinates relative to the output tile
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x > n || y > m) {
        return;
    }

    __shared__ float Qs[Tm * Tk];
    __shared__ float Cs[Tk * Tn];

    float out = 0.0f;
    float prev_max = 0.0f; // max logit of prev tile
    float prev_sum = 0.0f; // denominator of prev tile

    // Tiled loop over attn matrix cols
    for (int nn = 0; nn < n; nn += Tn) {

        float tmp = 0.0f;
        // Tiled loop over query cols
        for (int kk = 0; kk < k; kk += Tk) {

            Qs[threadIdx.y * Tk + threadIdx.x] = Q[y * k + kk + threadIdx.x];
            // Transposing C tile.  WARNING: shmem conflicts?
            Cs[threadIdx.x * Tm + threadIdx.y] =
                C[(nn + threadIdx.y) * k + kk + threadIdx.x];

            __syncthreads();

            for (int tk = 0; tk < Tk; ++tk) {
                tmp += Qs[threadIdx.y * Tk + tk] * Cs[tk * Tn + threadIdx.x];
            }

            __syncthreads();
        }

        tmp /= sqrtf(k);

        // Now tmp holds a tile of QC^T, I need to compute partial softmax

        // As long as Tm <= 32, use warp shuffles to compute partial reduction

        float max = tmp;
        // Butterfly reduction within half warp
        for (int s = 8; s > 0; s >>= 1) {
            max = fmaxf(max, __shfl_xor_sync(0xffffffff, max, s));
        }

        max = fmaxf(prev_max, max);

        float exp_log = expf(tmp - max);
        float sum = exp_log;
        for (int s = 8; s > 0; s >>= 1) {
            sum += __shfl_xor_sync(0xffffffff, sum, s);
        }

        float norm_coef = expf(prev_max - max);
        sum += prev_sum * norm_coef;
        out *= norm_coef;

        prev_max = max;
        prev_sum = sum;

        // Load exp logit to shmem, reuse Q
        Qs[threadIdx.y * Tn + threadIdx.x] = exp_log;
        Cs[threadIdx.y * Tk + threadIdx.x] = C[(nn + threadIdx.y) * k + x];

        __syncthreads();

        for (int tn = 0; tn < Tn; ++tn) {
            out += Qs[threadIdx.y * Tn + tn] * Cs[tn * Tk + threadIdx.x];
        }

        __syncthreads(); // Wait before reusing Qs and Cs
    }

    O[y * k + x] = out / prev_sum;
}

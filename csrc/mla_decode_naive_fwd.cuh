#pragma once

#include <cuda_fp16.h>

template<int block_size>
__global__ void mla_decode_naive_softmax(float *A, int n)
{
    float max = A[blockIdx.y * n + threadIdx.x];
    for (int j = blockDim.x + threadIdx.x; j < n; j += blockDim.x) {
        float val = A[blockIdx.y * n + j];
        if (val > max) {
            max = val;
        }
    }

    // TODO: Warp shuffle to reduce in warp

    __shared__ float tmp[block_size];
    tmp[threadIdx.x] = max;

    __syncthreads();

    for (int s = block_size >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (tmp[threadIdx.x + s] > tmp[threadIdx.x]) {
                tmp[threadIdx.x] = tmp[threadIdx.x + s];
            }
        }
        __syncthreads();
    }

    max = tmp[0];

    float sum = 0.0f;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        sum += expf(A[blockIdx.y * n + j] - max);
    }

    __syncthreads(); // Wait for max = tmp[0]

    tmp[threadIdx.x] = sum;

    __syncthreads();

    for (int s = block_size >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            tmp[threadIdx.x] += tmp[threadIdx.x + s];
        }
        __syncthreads();
    }

    sum = tmp[0];

    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        A[blockIdx.y * n + j] = expf(A[blockIdx.y * n + j] - max) / sum;
    }
}

template<int Tm, int Tn, int Tk, bool transb> // Whether B is transposed
__global__ void mla_decode_naive_gemm(const float *__restrict__ A,
                                      const float *__restrict__ B,
                                      float *__restrict__ O,
                                      int m,
                                      int n,
                                      int k,
                                      float alpha = 1.0f)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x > n || y > m) {
        return;
    }

    __shared__ float As[Tm * Tk];
    __shared__ float Bs[Tk * Tn];

    float tmp = 0.0f;
    for (int kk = 0; kk < k; kk += Tk) {

        // Assumes tiles are of the same size of the block
        As[threadIdx.y * Tk + threadIdx.x] = A[y * k + kk + threadIdx.x];

        if (transb) {
            // Transpose when storing in shmem
            Bs[threadIdx.x * Tn + threadIdx.y] =
                B[(blockDim.x * blockIdx.x + threadIdx.y) * k +
                    kk + threadIdx.x];
        } else {
            Bs[threadIdx.y * Tn + threadIdx.x] =
                B[(kk + threadIdx.y) * n + x];
        }

        __syncthreads();

        for (int tk = 0; tk < Tk; ++tk) {
            tmp += As[threadIdx.y * Tk + tk] * Bs[tk * Tn + threadIdx.x];
        }

        __syncthreads();
    }

    O[y * n + x] = alpha * tmp;
}

#pragma once

// TODO: Find a way to get around this

#ifdef __CUDA_NO_HALF_OPERATORS__
#undef __CUDA_NO_HALF_OPERATORS__
#endif

#ifdef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#endif

#include <cstdint>

#include <cuda_fp16.h>

#define WARP_SIZE 32

template<typename ScalarType>
struct KernelTraits {};

template<>
struct KernelTraits<__half>
{
    using Scalar = __half;

    static constexpr int BlockM = 32;
    static constexpr int BlockN = 16;

    static constexpr int HeadDimK = 576;
    static constexpr int HeadDimV = 512;

    static constexpr int NumWarps = 4;
    static constexpr int NumThreads = NumWarps * WARP_SIZE;

    static constexpr int MmaK = 16;

    static constexpr int VecLen = 8;
};

template<typename Scalar>
struct SharedMemory
{
    // TODO: Template on alloc alignment?
    __device__ __forceinline__ Scalar *alloc_tile(std::size_t size)
    {
        // Note that 128 byte alignment for shmem should increase
        // cp.async performance, but with 4 byte alignment the
        // throughput remains the same
        extern __shared__ __align__(128) unsigned char shmem[];

        Scalar *ptr = reinterpret_cast<Scalar *>(shmem + offset);

        offset += size * sizeof(Scalar);

        return ptr;
    }

private:
    uint64_t offset = 0;
};

template<typename Traits>
__device__ __forceinline__
void ldg_Q_tile(const void *__restrict__ Q_gmem,
                void *__restrict__ Q_shmem)
{
    #pragma unroll
    for (int i = 0;
             i < Traits::BlockM * Traits::HeadDimK / Traits::VecLen;
             i += Traits::NumThreads)
    {
        const int idx = i + threadIdx.x;

        reinterpret_cast<float4 *>(Q_shmem)[idx] =
            reinterpret_cast<const float4 *>(Q_gmem)[
                (Traits::BlockM * blockIdx.y) * Traits::HeadDimK /
                Traits::VecLen + idx];
    }
}

template<typename Traits>
__device__ __forceinline__
void cp_async_ldgsts_C_tile(const typename Traits::Scalar *__restrict__ C_gmem_src,
                            typename Traits::Scalar *__restrict__ C_shmem_dst)
{
    constexpr int VecLen = Traits::VecLen;
    constexpr int HeadDimK = Traits::HeadDimK;

    #pragma unroll
    for (int j = 0;
             j < Traits::BlockN * HeadDimK / VecLen;
             j += Traits::NumThreads) {

        const int idx = j + threadIdx.x;

        const typename Traits::Scalar *src = C_gmem_src + VecLen * idx;
        // Convert generic pointer to shared space state
        const uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(
            C_shmem_dst + VecLen * idx));

        asm volatile ("cp.async.cg.shared.global [%0], [%1], 16;"
                      :: "r"(dst), "l"(src));
    }
}

__device__ __forceinline__ void cp_async_commit()
{
    asm volatile ("cp.async.commit_group;");
}

template<int N>
__device__ __forceinline__ void cp_async_wait()
{
    asm volatile ("cp.async.wait_group %0;" :: "n"(N));
}

/*
 * - grid tiles:
 *                                                                                          512             64
 *                                                                               +-------------------------+--+
 *                                                                               |/////////////////////////|  |
 *                                                                               |/////////////////////////|  |
 *                                                    seq_len       C^t          |/////////////////////////|  |
 *                                     +-------------------------------------+   |/////////////////////////|  |  C
 *                                     |//////////////////|                  |   |/////////////////////////|  |
 *                                     |//////////////////|                  |   |/////////////////////////|  |
 *                                     |//////////////////|                  |   |/////////////////////////|  |
 *                                     |//////////////////|                  |   |-------------------------+  | seq_len
 *                                     |//////////////////|                  |   |                         |  |
 *                                576  |//////////////////|                  |   |                         |  |
 *                                     |//////////////////|                  |   |                         |  |
 *                                     |//////////////////|                  |   |                         |  |
 *                                     |//////////////////|                  |   |                         |  |
 *                                     |//////////////////|                  |   |                         |  |
 *                                     |//////////////////|                  |   |                         |  |
 *                                     +-------------------------------------+   +-------------------------+--+
 *                   Q
 *     +---------------------------+   +------------------+------------------+   +-------------------------+
 *     |///////////////////////////|   |///////cta0///////|       cta4       |   |//////////cta0///////////| 32
 *     |---------------------------|   +------------------+------------------+   |-------------------------|
 *     |                           |   |       cta1       |       cta5       |   |          cta1           |
 * 128 |---------------------------|   +------------------+------------------+   |-------------------------|  O
 *     |                           |   |       cta2       |       cta6       |   |          cta2           |
 *     |---------------------------|   +------------------+------------------+   |-------------------------|
 *     |                           |   |       cta3       |       cta7       |   |          cta3           |
 *     +---------------------------+   +------------------+------------------+   +-------------------------+
 *                  576                                   P                                   +
 *                                                                               +-------------------------+
 *                                                                               |          cta4           |
 *                                                                               |-------------------------|
 *                                                                               |          cta5           |
 *                                                                               |-------------------------|
 *                                                                               |          cta6           |
 *                                                                               |-------------------------|
 *                                                                               |          cta7           |
 *                                                                               +-------------------------+
 *
 *    Q is split vertically into 32x576 thread block tiles, C is split across the
 *    sequence lenght dimension to saturate all SMs (two splits is enough for 16 SMs).
 *    Note that P tiles are not materialized in gmem. Splits partial results
 *    are later reduced with a separate kernel.
 *
 *                                                          512
 * - thread block tiles:                            +--------------+-+
 *                                                  |//////////////| |Cs
 *                                                  +--------------+-+
 *                                                  |              | |
 *                        Cs       seq_len/2        |              | |
 *                       +--+---------------------+ |              | |
 *                       |//|                     | |              | | seq_len/2
 *                       |//|                     | |              | |
 *                  576  |//|     -->             | |              | |
 *                       |//|                     | |              | |
 *                       |//|                     | |              | |
 *            Qs         +--+---------------------+ +--------------+-+
 *    +----------------+ +--+---------------------+ +--------------+
 * 32 |////////////////| |//| Ps  -->             | |//////////////| Or
 *    +----------------+ +--+---------------------+ +--------------+
 *           576          16
 *
 *    each thread block is assigned a tile of Q that spans the entire head
 *    dimension and iterates over the tiles of a chunk of C.
 *
 * - warp tiles:
 *
 *                                                  16
 *                                                +----+
 *                                                |/w0/| 16
 *                                                +----+
 *                                                | w1 |
 *                                                +----+
 *                                                | w2 |
 *                                                +----+
 *                                                | w3 |
 *                                                +----+ Cs
 *                                                |/w0/|
 *                                                +----+
 *                                                | w1 |
 *                                                +----+
 *                                                | .. |                128           Cs
 *                                                +----+             +-------+-------+-------+-------+--+
 *                                                | w3 |             |///////|       |       |       |  |
 *                                                +----+             +-------+-------+-------+-------+--+
 * 
 *    +----+----+----+----+----+----+----+----+   +----+     +----+  +-------+-------+-------+-------+
 *    |/w0/| w1 | w2 | w3 |/w0/| w1 | .. | w3 |   |/w0/|  =  |////|  |/warp0/| warp1 | warp2 | warp3 |  Or
 * 32 |////|    |    |    |////|    |    |    |   |////|     |////|  |///////|       |       |       |
 *    +----+----+----+----+----+----+----+----+   +----+     +----+  +-------+-------+-------+-------+
 *      16               Qs                         +          Ps                  512  
 *                                                +----+
 *                                                | w1 | Pr
 *                                                |    |
 *                                                +----+
 *                                                  +
 *                                                +----+
 *                                                | w2 |
 *                                                |    |
 *                                                +----+
 *                                                  +
 *                                                +----+
 *                                                | w3 |
 *                                                |    |
 *                                                +----+
 *
 *    the QC^t product is performed as a slice-k gemm, each warp accumulates
 *    a partial product by performing a 32x144x16 gemm split across 9 32x16x16 slices,
 *    which is followed by a block wide reduction.
 *
 *
 * - mma tiles:
 *                     8    8                             8        128
 *                  +----+----+                        +----+----+----+----+----+
 *                  |////|    |                        |////|    |    |    |    |
 *         C_frag   |////|    |               C_frag   |////|    |    | .. |    |
 *                  |////|    |                        |////|    |    |    |    |
 *                  |////|    |                        |////|    |    |    |    |
 *                  +----+----+                        +----+----+----+----+----+
 *       Q_frag                               16
 *    +---------+   +----+----+          +---------+   +----+----+----+----+----+
 *    |/////////|   |////|    |          |/////////|   |////|    |    |    |    |
 * 16 |/////////|   |mma0|mma1|          |/////////|   |mma0|mma2|mma4| .. | mma|
 *    |/////////|   |////|    |          |/////////|   |////|    |    |    | 31 |
 *    |/////////|   |////|    |          |/////////|   |////|    |    |    |    |
 *    +---------+   +----+----+          +---------+   +----+----+----+----+----+
 *    |         |   |    |    |          |         |   |    |    |    |    |    |
 * 16 |         |   |mma2|mma3|          |         |   |mma1|mma3|mma5| .. | mma|
 *    |         |   |    |    |          |         |   |    |    |    |    | 32 |
 *    |         |   |    |    |          |         |   |    |    |    |    |    |
 *    +---------+   +----+----+          +---------+   +----+----+----+----+----+
 *        16            Pr                   Ps                   Or
 *
 *    each warp issues 4 m16n8k16 mma instructions per QC^t slice (36 mmas)
 *    and 32 m16n8k16 mma instructions for PC.
 *
 */

template<bool Trans = false>
__device__ __forceinline__ void ldmatrix_sync_m8n8_x2_b16(const float *src,
                                                          float frag[2])
{
    uint32_t *dst = reinterpret_cast<uint32_t *>(frag);
    uint32_t src_sh = static_cast<uint32_t>(__cvta_generic_to_shared(src));

    if constexpr (Trans) {
        asm volatile ("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];"
                      : "=r"(dst[0]), "=r"(dst[1])
                      : "r"(src_sh));
    } else {
        asm volatile ("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
                      : "=r"(dst[0]), "=r"(dst[1])
                      : "r"(src_sh));
    }
}

__device__ __forceinline__ void ldmatrix_sync_m8n8_x4_b16(const float *src,
                                                          float frag[4])
{
    uint32_t *dst = reinterpret_cast<uint32_t *>(frag);
    asm volatile ("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                 // threads 8-15 provide addr to top-right 8x8 matrix, which
                 // needs to be stored into {a4, a5}
                  : "=r"(dst[0]), "=r"(dst[2]), "=r"(dst[1]), "=r"(dst[3])
                  : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(src))));
}

__device__ __forceinline__ void mma_sync_m16n8k16_f16(const float A_frag[4],
                                                      const float B_frag[2],
                                                      const float C_frag[2],
                                                      float D_frag[2])
{
    // General purpose 32-bit registers are used as constraint
    // specifier for .f16x2 registers, so we need to cast them
    const uint32_t *A = reinterpret_cast<const uint32_t *>(A_frag);
    const uint32_t *B = reinterpret_cast<const uint32_t *>(B_frag);
    const uint32_t *C = reinterpret_cast<const uint32_t *>(C_frag);
    uint32_t *D = reinterpret_cast<uint32_t *>(D_frag);
    // row.col is the only layout possible on sm_89
    asm volatile ("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                  "{%0, %1}, "
                  "{%2, %3, %4, %5}, "
                  "{%6, %7}, "
                  "{%8, %9};"
                  : "=r"(D[0]), "=r"(D[1]) // dst
                  : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
                    "r"(B[0]), "r"(B[1]),
                    "r"(C[0]), "r"(C[1]));
}

__device__ __forceinline__ void stmatrix_m8n8_x2_b16(float *dst_start,
                                                     const int row_stride,
                                                     const float frag[2])
{
    // stmatrix instruction is not available on sm_89 :(, either we shuffle
    // and then use STS.128 or we stick to STS (32 bit)
    const int lane_idx = threadIdx.x % WARP_SIZE;
    // Coordinates of the fragment in each 8x8 matrix
    const int t_x = lane_idx % 4;
    const int t_y = lane_idx / 4;
    // Store upper and lower 8x8 fragment halves
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
        // Note that the given dst is not shifted by the row index as in ldmatrix
        dst_start[(h * 8 + t_y) * row_stride + t_x] = frag[h];
    }
}

template<typename KernelTraits>
__global__ void mla_decode_splitk_mma_fp16(const __half *__restrict__ Q,
                                           const __half *__restrict__ C,
                                           __half *__restrict__ O,
                                           __half *__restrict__ max,
                                           __half *__restrict__ sum,
                                           const int seq_length,
                                           const int num_splits)
{
    constexpr int BlockM = KernelTraits::BlockM;
    constexpr int BlockN = KernelTraits::BlockN;

    constexpr int HeadDimK = KernelTraits::HeadDimK;
    constexpr int HeadDimV = KernelTraits::HeadDimV;

    constexpr int NumWarps = KernelTraits::NumWarps;

    const int warp_idx = threadIdx.x / WARP_SIZE;
    const int lane_idx = threadIdx.x % WARP_SIZE;

    SharedMemory<__half> shmem;

    // TODO: There's no need to declare these as __half,
    // consider using float pointers for all buffers
    __half *__restrict__ Qs = shmem.alloc_tile(BlockM * HeadDimK);
    __half *__restrict__ Cs = shmem.alloc_tile(BlockN * HeadDimK * 2);
    __half *__restrict__ Ps = shmem.alloc_tile(BlockM * BlockN * NumWarps);

    ldg_Q_tile<KernelTraits>(Q, Qs);

    __syncthreads();

    // Double buffering for C
    __half *__restrict__ Cs_ld = Cs;
    __half *__restrict__ Cs_st = Cs + BlockN * HeadDimK;

    // Prefetch first C tile
    cp_async_ldgsts_C_tile<KernelTraits>(C, Cs_ld);
    cp_async_commit();

    const int split_seq_length = seq_length / num_splits;
    C += blockIdx.x * split_seq_length * HeadDimK;

    constexpr int NumMmasOutY = BlockM / 16;
    constexpr int NumMmasOutX = HeadDimV / (NumWarps * 8);

    float O_frag[NumMmasOutY * NumMmasOutX][2];

    // Loop over C tiles
    for (int C_j = 0; C_j < split_seq_length; C_j += BlockN) {

        // Prefetch next C tile
        if (C_j < split_seq_length - BlockN) {
            cp_async_ldgsts_C_tile<KernelTraits>(C + BlockN * HeadDimK, Cs_st);
        }
        cp_async_commit();

        cp_async_wait<1>(); // Wait for Cs_ld to be ready
        __syncthreads();

        constexpr int MmaK = KernelTraits::MmaK; // this has to match m16n8k16
        constexpr int NumSlicesPerWarp = HeadDimK / (NumWarps * MmaK);

        constexpr int NumMmasPerSliceX = BlockM / 16;
        constexpr int NumMmasPerSliceY = BlockN / 8;

        // Coordinates of the 8x8 sub-matrices, relative to a 16x16 matrix,
        // assigned to each thread when performing ldmatrix instructions
        const int mat_x = (lane_idx % 16) / 8;
        const int mat_y = lane_idx / 16;
        // Row within the 8x8 matrix
        const int mat_row = lane_idx % 8;

        // Accumulator for m16n8k16 mma has the same layout
        // as the left half of the first operand
        float P_frag[NumMmasPerSliceX * NumMmasPerSliceY][2];

        // Loop over the QC^t slices
        #pragma unroll
        for (int slice_idx = 0; slice_idx < NumSlicesPerWarp; ++slice_idx) {
            const int slice_offset = slice_idx * MmaK * NumWarps + warp_idx * MmaK;

            // Load Q_frag and C_frag from shmem to regs

            /*
             *  Each thread holds 4 32-bit values of the Q 16x16 slice
             *
             *      0   1   2   3   4   5   6   7   8   9   10  11  12  13  14  15
             *    +-------------------------------+-------------------------------+
             *  0 |t00 t00 t01 t01 t02 t02 t03 t03|t00 t00 t01 t01 t02 t02 t03 t03|
             *  1 |t04 t04 t05 t05 t06 t06 t07 t07|t04 t04 t05 t05 t06 t06 t07 t07|
             *    |             ...               |            ...                |
             *  7 |t28 t28 t29 t29 t30 t30 t31 t31|t28 t28 t29 t29 t30 t30 t31 t31|
             *    +-------------------------------+-------------------------------+
             *  8 |t00 t00 t01 t01 t02 t02 t03 t03|t00 t00 t01 t01 t02 t02 t03 t03|
             *  9 |t04 t04 t05 t05 t06 t06 t07 t07|t04 t04 t05 t05 t06 t06 t07 t07|
             *    |             ...               |            ...                |
             * 15 |t28 t28 t29 t29 t30 t30 t31 t31|t28 t28 t29 t29 t30 t30 t31 t31|
             *    +-------------------------------+-------------------------------+
             *
             *  and 2 32-bit values of the C^t 16x8 slice
             *
             *      0   1   ..   6   7
             *    +-------------------+
             *  0 |t00 t04     t25 t28|
             *  1 |t00 t04     t25 t28|
             *  2 |t01 t05     t25 t29|
             *  3 |t01 t05 ... t25 t29|
             *  4 |t02 t06     t26 t30|
             *  5 |t02 t06     t26 t30|
             *  6 |t02 t07     t27 t31|
             *  7 |t02 t07     t27 t31|
             *    +-------------------+
             *  8 |t00 t04     t25 t28|
             *  9 |t00 t04     t25 t28|
             * 10 |t01 t05     t25 t29|
             * 11 |t01 t05 ... t25 t29|
             * 12 |t02 t06     t26 t30|
             * 13 |t02 t06     t26 t30|
             * 14 |t02 t07     t27 t31|
             * 15 |t02 t07     t27 t31|
             *    +-------------------+
             */

            // Note that LDS.128 can't be used without shuffling since threads do
            // not own contiguous 16 bytes. That's probably why ldmatrix exists.

            float Q_frag[NumMmasPerSliceY][4]; // {a0, a1}, {a2, a3}, {a4, a5}, {a6, a7}
            float C_frag[NumMmasPerSliceX][2]; // {b0, b1}, {b2, b3}

            const float *Q_slice = // TODO: advance ptrs instead of adding the offset
                reinterpret_cast<const float *>(Qs + slice_offset);
            const float *C_slice =
                reinterpret_cast<const float *>(Cs_ld + slice_offset);

            // Each quarter warp loads one 8x8 matrix of the Q slice, and
            // each of the first two quarter warps load one 8x8 tile of the C slice

            // Explicitly staging loads before the fma loop proved to be slightly
            // better in the fp32 version. Perhaps for a large number of mma tiles
            // this is not ideal, but I guess ptxas would still manage to find a
            // proper reordering. This needs consideration when attempting to
            // pipeline shmem to regs loads, since one could try to pipeline within
            // the same slice or across slices, provided that ptxas agrees :/.
            // Perhaps in that case using membar.cta to force a specific ordering
            // becomes an option, since MIO throttling is less of an issue with
            // ldmatrix instructions.

            #pragma unroll
            for (int mma_y = 0; mma_y < NumMmasPerSliceY; ++mma_y) {
                // Each thread provides the ptr to one row of the 4 8x8 16 bit matrices
                ldmatrix_sync_m8n8_x4_b16(
                    Q_slice + (mma_y * 16 + mat_y * 8 + mat_row) *
                    HeadDimK / 2 + mat_x * 4, Q_frag[mma_y]);
            }

            #pragma unroll
            for (int mma_x = 0; mma_x < NumMmasPerSliceX; ++mma_x) {
                // mma expects the second operand to be in col major,
                // since we need C^t there's no need to transpose
                ldmatrix_sync_m8n8_x2_b16(C_slice + (mma_x * 8 + mat_row) *
                                          HeadDimK / 2 + mat_x * 4, C_frag[0]);
            }

            #pragma unroll
            for (int mma_y = 0; mma_y < NumMmasPerSliceY; ++mma_y) {
                #pragma unroll
                for (int mma_x = 0; mma_x < NumMmasPerSliceX; ++mma_x) {
                    mma_sync_m16n8k16_f16(Q_frag[mma_y], C_frag[mma_x],
                                          P_frag[mma_y * NumMmasPerSliceX + mma_x],
                                          P_frag[mma_y * NumMmasPerSliceX + mma_x]);
                }
            }
        }

        // Block reduce the accumulated slices

        float *__restrict__ Ps_warp =
            reinterpret_cast<float *>(Ps + warp_idx * BlockM * BlockN);

        // Store P_frag to shmem
        #pragma unroll
        for (int mma_y = 0; mma_y < NumMmasPerSliceY; ++mma_y) {
            #pragma unroll
            for (int mma_x = 0; mma_x < NumMmasPerSliceX; ++mma_x) {
                stmatrix_m8n8_x2_b16(Ps_warp + mma_y * 16 * BlockN / 2 +
                                     mma_x * 8 / 2, BlockN / 2,
                                     P_frag[mma_y * NumMmasPerSliceX + mma_x]);
            }
        }

        __syncthreads();

        // block_reduce_P_tile(Ps);
        // online_softmax();

        // Note that O_frag values have to be normalized before
        // accumulating the PC product

        { // Remove this inner scope and use a different name for P_frag
        float P_frag[NumMmasPerSliceY][4];

        // Load P_frag once, then iterate on the fragments of C and issue mmas,
        // consider N-stage pipelining or even prefetching in bulk

        #pragma unroll
        for (int mma_y = 0; mma_y < NumMmasOutY; ++mma_y) {
            // We're loading a 32x16 tile into registers with two 16x16 ldmatrix
            // instructions for Q as well, so this could be wrapped in a function
            ldmatrix_sync_m8n8_x4_b16(reinterpret_cast<float *>(Ps) +
                                      (mma_y * 16 + mat_y * 8 + mat_row) *
                                      BlockN / 2 + mat_x * 8 / 2,
                                      P_frag[mma_y]);
        }

        #pragma unroll // unroll factor may be relevant for pipelining
        for (int mma_x = 0; mma_x < NumMmasOutX; ++mma_x) {
            // Load C fragments, then issue corresponding mmas
            float *__restrict__ C_frag_src =
                reinterpret_cast<float *>(Cs_ld + warp_idx * NumMmasOutX * 8 +
                                          mma_x * 8) +
                    // threads 0-7 load upper half, 8-15 lower half,
                    // each pointing to a unique row of the 16x8 16-bit matrix
                    (mat_y * 8 + mat_row) * HeadDimK / 2;

            float C_frag[2];
            // Differently from before, we don't need C^t, so ldmatrix
            // has to transpose the fragments to bring them in col major
            ldmatrix_sync_m8n8_x2_b16(C_frag_src, C_frag);

            // WARNING: Something may be off with the previous ldmatrix,
            // the number of conflicts is unexpectedly low

            #pragma unroll
            for (int mma_y = 0; mma_y < NumMmasOutY; ++mma_y) {
                mma_sync_m16n8k16_f16(P_frag[mma_y], C_frag,
                                      O_frag[mma_y * NumMmasOutX + mma_x],
                                      O_frag[mma_y * NumMmasOutX + mma_x]);
            }
        }

        }

        // Swap Cs buffers
        __half *tmp = Cs_ld;
        Cs_ld = Cs_st;
        Cs_st = tmp;

        C += HeadDimK * BlockN;
    }

    // Store O_frag to shmem, then coalesced to gmem.
    // A single mma tile row is only half a cache line sector, moreover
    // we would be stuck with 32-bit wide gmem stores. Fully contiguous
    // 128 byte stores are still preferable than coalesced but non
    // contiguous 128 byte stores.

    // Qs can be reused safely to hold O fragments

    float *__restrict__ Os =
        reinterpret_cast<float *>(Qs + warp_idx * NumMmasOutX * 8);

    #pragma unroll
    for (int mma_y = 0; mma_y < NumMmasOutY; ++mma_y) {
        #pragma unroll
        for (int mma_x = 0; mma_x < NumMmasOutX; ++mma_x) {
            stmatrix_m8n8_x2_b16(Os + mma_y * 16 * HeadDimV / 2 +
                                 mma_x * 8 / 2, HeadDimV / 2,
                                 O_frag[mma_y * NumMmasOutX + mma_x]);
        }
    }
}

__global__ void mla_decode_combine()
{

}

void run_mla_decode_splitk_fp16(const __half *query,
                                const __half *cache,
                                __half *out,
                                __half *splits_max,
                                __half *splits_sum,
                                __half *splits_out,
                                int num_splits,
                                int num_heads,
                                int seq_length)
{
    using Traits = KernelTraits<__half>;

    dim3 block(Traits::NumThreads, 1);
    dim3 grid(num_splits, (num_heads - 1) / Traits::BlockM + 1);

    int shmem_bytes = (Traits::BlockM * Traits::HeadDimK +
                       Traits::BlockN * Traits::HeadDimK  * 2 +
                       Traits::BlockM * Traits::BlockN * Traits::NumWarps) *
                      sizeof(__half);

    // Dynamic shmem is required for allocations > 48KB
    cudaError_t error = cudaFuncSetAttribute(
        mla_decode_splitk_mma_fp16<Traits>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shmem_bytes);

    if (error != cudaSuccess) {
        printf("Error: %s (%s)\n",
               cudaGetErrorName(error),
               cudaGetErrorString(error));
    }

    mla_decode_splitk_mma_fp16<Traits><<<grid, block, shmem_bytes>>>(
        query, cache, splits_out, splits_max, splits_sum,
        seq_length, num_splits);

    //mla_decode_combine<>

    if ((error = cudaGetLastError()) != cudaSuccess) {
        printf("Error: %s (%s)\n",
               cudaGetErrorName(error),
               cudaGetErrorString(error));
    }
}

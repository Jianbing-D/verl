#ifndef CSRC_LIBRARY_SM80_H
#define CSRC_LIBRARY_SM80_H

#include <cstdint>
#include <cute/tensor.hpp>

namespace lce {

template <int32_t _M, int32_t _N, int32_t _K>
struct _3DLayout {
    static constexpr int32_t M = _M;
    static constexpr int32_t N = _N;
    static constexpr int32_t K = _K;
};

using namespace cute;

template <typename InT, typename OutT,
          int32_t _dim,
          typename _ThreadLayout = _3DLayout<4, 1, 1>>
struct Traits {
    using IN_DTYPE = InT;
    using OUT_DTYPE = OutT;
    static_assert(std::is_same_v<IN_DTYPE, OUT_DTYPE>, "IN_DTYPE and OUT_DTYPE must be the same");

    using ThreadLayout = _ThreadLayout;
    static_assert(ThreadLayout::K == 1, "ThreadLayout::K must be 1");

    static constexpr int32_t tileM = 64;
    static constexpr int32_t tileN = 64;
    static constexpr int32_t tileK = 128 / sizeof(InT);

    // length of elements per token, that is hidden_size
    static constexpr int32_t dim = _dim;
    static_assert(dim % tileK == 0, "dim must be divisible by tileK");


    // pipeline size
    static constexpr int32_t pipe = 3;

    // specify the basic MMA instruction
    using MMA_INST = SM80_16x8x16_F32BF16BF16F32_TN;
    using MMA_ATOM = MMA_Atom<MMA_INST>;
    using MMA_ATOM_TRAITS = MMA_Traits<MMA_INST>;

    // LDGSTS.128, which will handle 128 bytes of data per access
    using CopyAtom = Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, InT>;

    // LDSM
    using LdsmAtom = Copy_Atom<SM75_U32x4_LDSM_N, InT>;
    using LdsmTransAtom = Copy_Atom<SM75_U16x8_LDSM_T, InT>;

    // Tiled MMA
    static constexpr int32_t m_per_wave = get<0>(MMA_ATOM_TRAITS::Shape_MNK{}) * ThreadLayout::M;
    static constexpr int32_t n_per_wave = get<1>(MMA_ATOM_TRAITS::Shape_MNK{}) * ThreadLayout::N;
    static constexpr int32_t k_per_wave = get<2>(MMA_ATOM_TRAITS::Shape_MNK{}) * ThreadLayout::K;
    static constexpr int32_t m_waves = tileM / m_per_wave;
    static constexpr int32_t n_waves = tileN / n_per_wave;
    static constexpr int32_t k_waves = tileK / k_per_wave;
    using TiledMMA = TiledMMA<MMA_ATOM,
                             Layout<Shape<Int<ThreadLayout::M>, Int<ThreadLayout::N>, _1>>,
                             Tile<Int<get<0>(MMA_ATOM_TRAITS::Shape_MNK{}) * ThreadLayout::M>,
                                  Int<get<1>(MMA_ATOM_TRAITS::Shape_MNK{}) * ThreadLayout::N>,
                                  Int<get<2>(MMA_ATOM_TRAITS::Shape_MNK{}) * ThreadLayout::K>>>;
    static constexpr int32_t threads = get<0>(shape(MMA_ATOM_TRAITS::ThrID{})) 
                                                  * ThreadLayout::M
                                                  * ThreadLayout::N
                                                  * ThreadLayout::K;
    static_assert(threads == size(TiledMMA{}), "threads must be equal to the size of TiledMMA");
    static_assert(threads <= 1024, "threads must be less than or equal to 1024");

    // Tiled Copy
    static constexpr int32_t num_elems_per_chunk = sizeof(uint128_t) / sizeof(InT);
    static constexpr int32_t _copy_thr_per_row = tileK / num_elems_per_chunk;
    // all threads participated in the LDGSTS
    using TiledCopy = decltype(make_tiled_copy(CopyAtom{},
                                               /*thr*/Layout<Shape<Int<threads / _copy_thr_per_row>,
                                                                   Int<_copy_thr_per_row>>,
                                                             Stride<Int<_copy_thr_per_row>, _1>>{},
                                               /*value=*/Layout<Shape<_1, Int<num_elems_per_chunk>>>{}));

    // make sure all threads participate in the LDGSTS
    static_assert(size(TiledCopy{}) == threads, "all threads must participate in the LDGSTS");

    // the swizzle pattern is 128B, and repeated in 1024B
    static constexpr int32_t chunk_size_bits = std::is_same_v<IN_DTYPE, float> ? 2 : 3;
    using SmemLayoutAtom = decltype(composition(
        Swizzle<3, chunk_size_bits, 3>{},
        Layout<Shape<_8, Int<tileK>>, 
               Stride<Int<tileK>, _1>>{}));

    using SmemLayoutHidden = decltype(tile_to_shape(SmemLayoutAtom{},
                                                    Shape<Int<tileM>, Int<tileK>, Int<pipe>>{}));
    using SmemLayoutWeight = decltype(tile_to_shape(SmemLayoutAtom{},
                                                    Shape<Int<tileN>, Int<tileK>, Int<pipe>>{}));
    static constexpr size_t smem_hidden_bytes = sizeof(InT) * size(SmemLayoutHidden{});
    static constexpr size_t smem_weight_bytes = sizeof(InT) * size(SmemLayoutWeight{}); 

    static constexpr size_t smem_bytes = smem_hidden_bytes + smem_weight_bytes
                                          + 1024; // additional 1024 bytes for alignment
};


template <typename Traits>
__global__ void forward_mainloop_kernel(int32_t rank,
                                        typename Traits::IN_DTYPE *hidden_ptr,
                                        int32_t stride_hidden_m, int32_t stride_hidden_k,
                                        typename Traits::IN_DTYPE *weight_ptr,
                                        int32_t stride_weight_n, int32_t stride_weight_k,
                                        uint64_t *labels_ptr,
                                        int32_t num_tokens,
                                        int32_t vocab_size,
                                        int32_t vocab_per_split,
                                        float *gmem_output_ptr) {
    extern __shared__ char smem_[];
    char *smem_aligned = (char*)(((intptr_t)smem_ + 1023) & ~1023);
    (void)smem_aligned;

    int32_t m_tile_id = blockIdx.x;

    Tensor mHidden = make_tensor(make_gmem_ptr(reinterpret_cast<typename Traits::IN_DTYPE*>(hidden_ptr)),
                                 make_shape(num_tokens, Int<Traits::dim>{}),
                                 make_stride(Int<Traits::dim>{}, _1{}));
    Tensor mWeight = make_tensor(make_gmem_ptr(reinterpret_cast<typename Traits::IN_DTYPE*>(weight_ptr)),
                                 make_shape(vocab_size, Int<Traits::dim>{}),
                                 make_stride(Int<Traits::dim>{}, _1{}));
#if 1
    Tensor mC = make_tensor(make_gmem_ptr(reinterpret_cast<float*>(gmem_output_ptr)),
                            make_shape(num_tokens, vocab_size),
                            make_stride(vocab_size, _1{}));
#endif


    // [(tileM, tileK), k]
    Tensor gHidden = local_tile(mHidden, 
                                Shape<Int<Traits::tileM>, Int<Traits::tileK>>{},
                                make_coord(m_tile_id, _));
    // [(tileN, tileK), n, k]
    Tensor gWeight = local_tile(mWeight,
                                Shape<Int<Traits::tileN>, Int<Traits::tileK>>{},
                                make_coord(_, _));
#if 1
    // [(tileM, tileN), 1, n]
    Tensor gC = local_tile(mC,
                           Shape<Int<Traits::tileM>, Int<Traits::tileN>>{},
                           make_coord(m_tile_id, _));
#endif

#if 0
    // Print gHidden's shape
    if (m_tile_id == 0 && threadIdx.x == 0) {
        printf("gHidden: (%d, %d, %d)\n",
               (int32_t)size<0>(gHidden),
               (int32_t)size<1>(gHidden),
               (int32_t)size<2>(gHidden));

        printf("gWeight: (%d, %d, %d, %d)\n",
               (int32_t)size<0>(gWeight),
               (int32_t)size<1>(gWeight),
               (int32_t)size<2>(gWeight),
               (int32_t)size<3>(gWeight));
    }
#endif
    // (tileM, tileK, pipe)
    Tensor sHidden = make_tensor(make_smem_ptr(reinterpret_cast<typename Traits::IN_DTYPE*>(smem_aligned)),
                                 typename Traits::SmemLayoutHidden{});
    // (tileN, tileK, pipe)
    Tensor sWeight = make_tensor(make_smem_ptr(reinterpret_cast<typename Traits::IN_DTYPE*>(smem_aligned
                                        + Traits::smem_hidden_bytes)),
                                 typename Traits::SmemLayoutWeight{});
    
#if 0
    if (m_tile_id == 0 && threadIdx.x == 0) {
        printf("sHidden: (%d, %d, %d)\n",
               (int32_t)size<0>(sHidden),
               (int32_t)size<1>(sHidden),
               (int32_t)size<2>(sHidden));

        printf("sWeight: (%d, %d, %d)\n",
               (int32_t)size<0>(sWeight),
               (int32_t)size<1>(sWeight),
               (int32_t)size<2>(sWeight));
    }
#endif
    // GMEM -> SMEM
    typename Traits::TiledCopy gmem_tiled_copy_hidden;
    typename Traits::TiledCopy gmem_tiled_copy_weight;
    ThrCopy thr_copy_hidden = gmem_tiled_copy_hidden.get_slice(threadIdx.x);
    ThrCopy thr_copy_weight = gmem_tiled_copy_weight.get_slice(threadIdx.x);
    
    // GMEM hidden in thread A layout
    Tensor tAgH = thr_copy_hidden.partition_S(gHidden); // (CPY, CPY_M, CPY_K, k)
    Tensor tAsH = thr_copy_hidden.partition_D(sHidden); // (CPY, CPY_M, CPY_K, pipe)
    // GMEM weight in thread B layout
    Tensor tBgW = thr_copy_weight.partition_S(gWeight); // (CPY, CPY_N, CPY_K, n, k)
    Tensor tBsW = thr_copy_weight.partition_D(sWeight); // (CPY, CPY_N, CPY_K, pipe)

    static_assert(size<0>(tAgH) == size<0>(tAsH), "tAgH and tAsH must have the same elements per copy");
    static_assert(size<1>(tAgH) == size<1>(tAsH), "tAgH and tAsH must have the same number of copies along M");
    static_assert(size<2>(tAgH) == size<2>(tAsH), "tAgH and tAsH must have the same number of copies along K");
    static_assert(size<0>(tBgW) == size<0>(tBsW), "tBgW and tBsW must have the same elements per copy");
    static_assert(size<1>(tBgW) == size<1>(tBsW), "tBgW and tBsW must have the same number of copies along N");
    static_assert(size<2>(tBgW) == size<2>(tBsW), "tBgW and tBsW must have the same number of copies along K");

    // predicate
    Tensor cH = make_identity_tensor(make_shape(size<0>(sHidden), size<1>(sHidden)));
    Tensor cW = make_identity_tensor(make_shape(size<0>(sWeight), size<1>(sWeight)));
    Tensor tAcH = thr_copy_hidden.partition_S(cH); // (CPY, CPY_M, CPY_K)
    Tensor tBcW = thr_copy_weight.partition_S(cW); // (CPY, CPY_N, CPY_K)
    // predicate for hidden along M
    Tensor tApH = make_tensor<bool>(make_shape(size<1>(tAsH))); // (CPY_M)
    CUTLASS_PRAGMA_UNROLL
    for (int32_t m = 0; m < size<0>(tApH); ++m) {
        tApH(m) = get<0>(tAcH(0, m, 0)) < num_tokens - m_tile_id * Traits::tileM;
    }
    // predicate for weight along N
    Tensor tBpW = make_tensor<bool>(make_shape(size<1>(tBsW))); // (CPY_N)
#if 0
    if (m_tile_id == 0 && threadIdx.x == 0) {
        printf("tAgH: (%d, %d, %d, %d)\n",
               (int32_t)size<0>(tAgH),
               (int32_t)size<1>(tAgH),
               (int32_t)size<2>(tAgH),
               (int32_t)size<3>(tAgH));

        printf("tAsH: (%d, %d, %d, %d)\n",
               (int32_t)size<0>(tAsH),
               (int32_t)size<1>(tAsH),
               (int32_t)size<2>(tAsH),
               (int32_t)size<3>(tAsH));

        printf("tBgW: (%d, %d, %d, %d, %d)\n",
               (int32_t)size<0>(tBgW),
               (int32_t)size<1>(tBgW),
               (int32_t)size<2>(tBgW),
               (int32_t)size<3>(tBgW),
               (int32_t)size<4>(tBgW));

        printf("tBsW: (%d, %d, %d, %d)\n",
               (int32_t)size<0>(tBsW),
               (int32_t)size<1>(tBsW),
               (int32_t)size<2>(tBsW),
               (int32_t)size<3>(tBsW));

        print("tAgH layout: "); print(tAgH); print("\n");
        print("tAcH layout: "); print(tAcH); print("\n");
        print("tAcH[0, 0, 0]: "); print(tAcH(0, 0, 0)); print("\n");
        print("get<0>(tAcH(0, 0, 0)): "); print(get<0>(tAcH(0, 0, 0))); print("\n");
        print("tAcH[0, 1, 0]: "); print(tAcH(0, 1, 0)); print("\n");
        print("get<0>(tAcH(0, 1, 0)): "); print(get<0>(tAcH(0, 1, 0))); print("\n");
    }
#endif

    typename Traits::TiledMMA tiled_mma;
    ThrMMA thr_mma = tiled_mma.get_slice(threadIdx.x);
    Tensor tCsH = thr_mma.partition_A(sHidden); // (MMA, MMA_M, MMA_K, pipe)
    Tensor tCsW = thr_mma.partition_B(sWeight); // (MMA, MMA_N, MMA_K, pipe)

    Tensor cC = make_identity_tensor(make_shape(size<0>(gC), size<1>(gC)));
    Tensor tCcC = thr_mma.partition_C(cC); // (MMA, MMA_M, MMA_N)

#if 1
    Tensor tCgC = thr_mma.partition_C(gC); // (MMA, MMA_M, MMA_N, n)
#endif

    // allocate the accumulator, (MMA, MMA_M, MMA_N)
    Tensor tCrO = partition_fragment_C(tiled_mma, 
                                        Shape<Int<Traits::tileM>, Int<Traits::tileN>>{});

    // allocate register for pipelining
    Tensor tCrH = thr_mma.make_fragment_A(tCsH(_, _, _, 0));
    Tensor tCrW = thr_mma.make_fragment_B(tCsW(_, _, _, 0));

#if 0
    if (m_tile_id == 0 && threadIdx.x == 0) {
        printf("tCsH: (%d, %d, %d, %d)\n",
               (int32_t)size<0>(tCsH),
               (int32_t)size<1>(tCsH),
               (int32_t)size<2>(tCsH),
               (int32_t)size<3>(tCsH));

        printf("tCrO: (%d, %d, %d)\n",
               (int32_t)size<0>(tCrO),
               (int32_t)size<1>(tCrO),
               (int32_t)size<2>(tCrO));

        print("tCsH layout: "); print(tCsH); print("\n");
    }
#endif


    int32_t n_tile_count = size<3>(tBgW);
    for (int32_t n_tile = 0; n_tile < n_tile_count; ++n_tile) {
        // update the predicate for weight along N
        CUTLASS_PRAGMA_UNROLL
        for (int32_t n = 0; n < size<0>(tBpW); ++n) {
            tBpW(n) = get<0>(tBcW(0, n, 0)) < vocab_size - n_tile * Traits::tileN;
        }

        // Total count of tiles along K
        int32_t k_tile_count = Int<(int32_t)size<3>(tAgH)>::value;
        // current tile index in GMEM to read from
        int32_t k_tile_next = 0;

        #pragma unroll
        for (int32_t k_pipe = 0; k_pipe < Traits::pipe - 1; ++k_pipe) {
            #pragma unroll
            for (int32_t m = 0; m < size<1>(tAgH); ++m) {
                if (tApH(m)) {
                    copy(gmem_tiled_copy_hidden,
                         /*src=*/tAgH(_, m, _, k_tile_next),
                         /*dst=*/tAsH(_, m, _, k_pipe));
                }
            }
            #pragma unroll
            for (int32_t n = 0; n < size<1>(tBgW); ++n) {
                if (tBpW(n)) {
                    copy(gmem_tiled_copy_weight,
                         /*src=*/tBgW(_, n, _, n_tile, k_tile_next),
                         /*dst=*/tBsW(_, n, _, k_pipe));
                }
            }
            cp_async_fence();
            --k_tile_count;
            if (k_tile_count > 0) { ++k_tile_next; }
        }

        clear(tCrO);

        // current pipe in smem to read from
        int32_t smem_pipe_read = 0;
        int32_t smem_pipe_write = Traits::pipe - 1;

        Tensor tCsH_p = tCsH(_, _, _, smem_pipe_read); // (MMA, MMA_M, MMA_K)
        Tensor tCsW_p = tCsW(_, _, _, smem_pipe_read); // (MMA, MMA_N, MMA_K)

        // prefetch first k-block
        constexpr int32_t k_block_max = size<2>(tCrH);
        if constexpr (k_block_max > 1) {
            cp_async_wait<Traits::pipe - 2>();
            __syncthreads();

            // SMEM -> REG
            copy(/*src=*/tCsH_p(_, _, Int<0>{}), 
                 /*dst=*/tCrH(_, _, Int<0>{}));
            copy(/*src=*/tCsW_p(_, _, Int<0>{}),
                /*dst=*/tCrW(_, _, Int<0>{}));
        }

        // pipelined main loop on k-tiles
        CUTE_NO_UNROLL
        while (k_tile_count > -(Traits::pipe - 1)) {
            // iterate over k-blocks
            CUTE_UNROLL
            for (int32_t k = 0; k < k_block_max; ++k) {
                // wait for k-tile to be loaded into SMEM
                if (k == k_block_max - 1) {
                    tCsH_p = tCsH(_, _, _, smem_pipe_read);
                    tCsW_p = tCsW(_, _, _, smem_pipe_read);
                    cp_async_wait<Traits::pipe - 2>();
                    __syncthreads();
                }

                // SMEM -> REG for k+1
                int32_t k_next = (k + Int<1>{}) % k_block_max;
                copy(tCsH_p(_,_,k_next), tCrH(_,_,k_next));
                copy(tCsW_p(_,_,k_next), tCrW(_,_,k_next));

                // launch GMEM -> SMEM
                if (k == 0) {
                    #pragma unroll
                    for (int32_t m = 0; m < size<1>(tAgH); ++m) {
                        if (tApH(m)) {
                            copy(gmem_tiled_copy_hidden,
                                 tAgH(_, m, _, k_tile_next),
                                 tAsH(_, m, _, smem_pipe_write));
                        }
                    }
                    #pragma unroll
                    for (int32_t n = 0; n < size<1>(tBgW); ++n) {
                        if (tBpW(n)) {
                            copy(gmem_tiled_copy_weight,
                                 tBgW(_, n, _, n_tile, k_tile_next),
                                 tBsW(_, n, _, smem_pipe_write));
                        }
                    }
                    cp_async_fence();

                    // advance GMEM tile
                    --k_tile_count;
                    if (k_tile_count > 0) { ++k_tile_next; }

                    // advance SMEM pipe
                    smem_pipe_write = smem_pipe_read;
                    ++smem_pipe_read;
                    smem_pipe_read = (smem_pipe_read == Traits::pipe) ? 0 : smem_pipe_read;
                }

                // MMA
                gemm(tiled_mma, /*A*/tCrH(_, _, k), /*B*/tCrW(_, _, k), /*C*/tCrO);
            } // iterate over k-blocks
        } // iterate over k-tiles

        // epilogue on this n-tile
    #if 1
        // REG -> GMEM
        if (gmem_output_ptr != nullptr) {
            #pragma unroll
            for (int32_t k = 0; k < size<0>(tCcC); ++k) {
                #pragma unroll
                for (int32_t m = 0; m < size<1>(tCcC); ++m) {
                    #pragma unroll 
                    for (int32_t n = 0; n < size<2>(tCcC); ++n) {
                        if (get<0>(tCcC(k, m, n)) < num_tokens - m_tile_id * Traits::tileM
                            && get<1>(tCcC(k, m, n)) < vocab_size - n_tile * Traits::tileN) {
                            tCgC(k, m, n, n_tile) = tCrO(k, m, n);
                        }
                    }
                }
            }
        }
    #endif

    } // n_tile
}

} // namespace lce
#endif


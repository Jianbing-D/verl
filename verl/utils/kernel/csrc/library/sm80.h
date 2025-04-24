#ifndef CSRC_LIBRARY_SM80_H
#define CSRC_LIBRARY_SM80_H

#include <cstdint>
#include <cute/tensor.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace lce {

#define _ENABLE_GMEM_RESULT 0

namespace cg = cooperative_groups;

template <int32_t _M, int32_t _N, int32_t _K>
struct _3DLayout {
    static constexpr int32_t M = _M;
    static constexpr int32_t N = _N;
    static constexpr int32_t K = _K;
};

using namespace cute;

template <typename InT, typename OutT,
          int32_t _dim,
          typename _ThreadLayout = _3DLayout<4, 2, 1>>
struct Traits {
    using IN_DTYPE = InT;
    using OUT_DTYPE = OutT;
    static_assert(std::is_same_v<IN_DTYPE, OUT_DTYPE>, "IN_DTYPE and OUT_DTYPE must be the same");

    using ThreadLayout = _ThreadLayout;
    static_assert(ThreadLayout::K == 1, "ThreadLayout::K must be 1");

    static constexpr int32_t tileM = 128;
    static constexpr int32_t tileN = 128;
    static constexpr int32_t tileK = 128 / sizeof(InT);

    // length of elements per token, that is hidden_size
    static constexpr int32_t dim = _dim;
    static_assert(dim % tileK == 0, "dim must be divisible by tileK");

    static constexpr int32_t threadBlockSwizzleSize = 4;

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
                                  Int<get<1>(MMA_ATOM_TRAITS::Shape_MNK{}) * 2 * ThreadLayout::N>,
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

    using TiledLDSM_A = decltype(make_tiled_copy_A(LdsmAtom{}, TiledMMA{}));
    using TiledLDSM_B = decltype(make_tiled_copy_B(LdsmAtom{}, TiledMMA{}));

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

    using SmemLayoutOutput = decltype(tile_to_shape(SmemLayoutAtom{},
                                                    Shape<Int<tileM>, Int<tileN>>{}));
    static constexpr size_t smem_output_bytes = sizeof(float) * size(SmemLayoutOutput{});

    using SmemLinearLayout = Layout<Shape<Int<tileM>, _1>,
                                    Stride<_1, Int<tileM>>>;
    static constexpr size_t smem_labels_bytes = sizeof(int64_t) * size(SmemLinearLayout{});

    static constexpr size_t smem_bytes = smem_hidden_bytes 
                                        + smem_weight_bytes 
                                        + smem_output_bytes
                                        + smem_labels_bytes
                                        + 1024; // additional 1024 bytes for alignment
};


template <typename Traits>
__launch_bounds__(Traits::threads)
__global__ void forward_mainloop_kernel(int32_t rank,
                                        typename Traits::IN_DTYPE *hidden_ptr,
                                        typename Traits::IN_DTYPE *weight_ptr,
                                        int64_t *labels_ptr,
                                        int32_t num_tokens,
                                        int32_t vocab_size,
                                        int32_t vocab_per_split,
                                        int32_t num_splits,
                                        float *max_ptr,
                                        float *acc_ptr,
                                        float *entropy_b_ptr,
                                        float *logprobs_ptr,
                                        float *gmem_output_ptr) {
    extern __shared__ char smem_[];
    char *smem_aligned = (char*)(((intptr_t)smem_ + 1023) & ~1023);

    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);

    int32_t num_pid_m = (num_tokens + Traits::tileM - 1) / Traits::tileM;
    int32_t num_pid_n = (vocab_size + vocab_per_split - 1) / vocab_per_split;

    const auto [pid_m, pid_n] = [&] () -> std::tuple<int32_t, int32_t> {
        // int32_t num_pid_m = (num_tokens + Traits::tileM - 1) / Traits::tileM;
        // int32_t pid_m = blockIdx.x % num_pid_m;
        // int32_t pid_n = blockIdx.x / num_pid_m;
        // return {pid_m, pid_n};
        int32_t gdimx = num_pid_m * Traits::threadBlockSwizzleSize;
        int32_t bidm = blockIdx.x % gdimx;
        int32_t bidn = blockIdx.x / gdimx;
        int32_t pid_m = bidm / Traits::threadBlockSwizzleSize;
        int32_t pid_n = bidn * Traits::threadBlockSwizzleSize + (bidm % Traits::threadBlockSwizzleSize);
        return {pid_m, pid_n};
    }();
    if (pid_m >= num_pid_m || pid_n >= num_pid_n) { return; }

    Tensor mHidden = make_tensor(make_gmem_ptr(reinterpret_cast<typename Traits::IN_DTYPE*>(hidden_ptr)),
                                 make_shape(num_tokens, Int<Traits::dim>{}),
                                 make_stride(Int<Traits::dim>{}, _1{}));
    // (vocab_size, dim)
    Tensor mWeight = make_tensor(make_gmem_ptr(reinterpret_cast<typename Traits::IN_DTYPE*>(weight_ptr)),
                                 make_shape(vocab_size, Int<Traits::dim>{}),
                                 make_stride(Int<Traits::dim>{}, _1{}));

    // (vocab_per_split, dim)
    Tensor mWeight_n = local_tile(mWeight,
                                  make_shape(vocab_per_split, Int<Traits::dim>{}),
                                  make_coord(pid_n, Int<0>{}));
    // (num_tokens, 1)
    Tensor mLabels = make_tensor(make_gmem_ptr(reinterpret_cast<int64_t*>(labels_ptr)),
                                 make_shape(num_tokens, _1{}));

    // (num_tokens, num_splits)
    Tensor mMax = make_tensor(make_gmem_ptr(reinterpret_cast<float*>(max_ptr)),
                              make_shape(num_tokens, num_splits),
                              make_stride(num_splits, _1{}));
    Tensor mAcc = make_tensor(make_gmem_ptr(reinterpret_cast<float*>(acc_ptr)),
                              make_shape(num_tokens, num_splits),
                              make_stride(num_splits, _1{}));
    Tensor mEntropyB = make_tensor(make_gmem_ptr(reinterpret_cast<float*>(entropy_b_ptr)),
                                   make_shape(num_tokens, num_splits),
                                   make_stride(num_splits, _1{}));

    // (num_tokens, 1)
    Tensor mLogprobs = make_tensor(make_gmem_ptr(reinterpret_cast<float*>(logprobs_ptr)),
                                   make_shape(num_tokens, _1{}));

#if _ENABLE_GMEM_RESULT == 1
    // (num_tokens, vocab_size)
    Tensor mC = make_tensor(make_gmem_ptr(reinterpret_cast<float*>(gmem_output_ptr)),
                            make_shape(num_tokens, vocab_size),
                            make_stride(vocab_size, _1{}));
    // (num_tokens, vocab_per_split)
    Tensor mC_n = local_tile(mC,
                            make_shape(num_tokens, vocab_per_split),
                            make_coord(Int<0>{}, pid_n));
#endif

    // [(tileM, tileK), k]
    Tensor gHidden = local_tile(mHidden, 
                                Shape<Int<Traits::tileM>, Int<Traits::tileK>>{},
                                make_coord(pid_m, _));
    // [(tileN, tileK), n, k]
    Tensor gWeight = local_tile(mWeight_n,
                                Shape<Int<Traits::tileN>, Int<Traits::tileK>>{},
                                make_coord(_, _));
    // (tileM, 1)
    Tensor gLabels = local_tile(mLabels,
                                Shape<Int<Traits::tileM>, _1>{},
                                make_coord(pid_m, Int<0>{}));

    // (tileM, 1)
    Tensor gMax = local_tile(mMax,
                             Shape<Int<Traits::tileM>, _1>{},
                             make_coord(pid_m, pid_n));
    Tensor gAcc = local_tile(mAcc,
                             Shape<Int<Traits::tileM>, _1>{},
                             make_coord(pid_m, pid_n));
    Tensor gEntropyB = local_tile(mEntropyB,
                                   Shape<Int<Traits::tileM>, _1>{},
                                   make_coord(pid_m, pid_n));

    // (tileM, 1)
    Tensor gLogprobs = local_tile(mLogprobs,
                                  Shape<Int<Traits::tileM>, _1>{},
                                  make_coord(pid_m, Int<0>{}));

#if _ENABLE_GMEM_RESULT == 1
    // [(tileM, tileN), 1, n]
    Tensor gC = local_tile(mC_n,
                           Shape<Int<Traits::tileM>, Int<Traits::tileN>>{},
                           make_coord(pid_m, _));
#endif

    // (tileM, tileK, pipe)
    Tensor sHidden = make_tensor(make_smem_ptr(reinterpret_cast<typename Traits::IN_DTYPE*>(smem_aligned)),
                                 typename Traits::SmemLayoutHidden{});
    // (tileN, tileK, pipe)
    Tensor sWeight = make_tensor(make_smem_ptr(reinterpret_cast<typename Traits::IN_DTYPE*>(smem_aligned
                                        + Traits::smem_hidden_bytes)),
                                 typename Traits::SmemLayoutWeight{});
    // (tileM, tileN)
    Tensor sLogit = make_tensor(make_smem_ptr(reinterpret_cast<float*>(smem_aligned
                                        + Traits::smem_hidden_bytes
                                        + Traits::smem_weight_bytes)),
                                 typename Traits::SmemLayoutOutput{});
    // (tileM, 1)
    Tensor sLabels = make_tensor(make_smem_ptr(reinterpret_cast<int64_t*>(smem_aligned
                                        + Traits::smem_hidden_bytes
                                        + Traits::smem_weight_bytes
                                        + Traits::smem_output_bytes)),
                                 typename Traits::SmemLinearLayout{});    
                                
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
        tApH(m) = get<0>(tAcH(0, m, 0)) < num_tokens - pid_m * Traits::tileM;
    }
    // predicate for weight along N
    Tensor tBpW = make_tensor<bool>(make_shape(size<1>(tBsW))); // (CPY_N)

    typename Traits::TiledMMA tiled_mma;
    ThrMMA thr_mma = tiled_mma.get_slice(threadIdx.x);
    Tensor tCsH = thr_mma.partition_A(sHidden); // (MMA, MMA_M, MMA_K, pipe)
    Tensor tCsW = thr_mma.partition_B(sWeight); // (MMA, MMA_N, MMA_K, pipe)

    Tensor cC = make_identity_tensor(make_shape(size<0>(sLogit), size<1>(sLogit)));
#if _ENABLE_GMEM_RESULT == 1
    Tensor tCcC = thr_mma.partition_C(cC); // (MMA, MMA_M, MMA_N)
    Tensor tCgC = thr_mma.partition_C(gC); // (MMA, MMA_M, MMA_N, n)
#endif

    // allocate the accumulator, (MMA, MMA_M, MMA_N)
    Tensor tCrO = partition_fragment_C(tiled_mma, 
                                        Shape<Int<Traits::tileM>, Int<Traits::tileN>>{});

    // allocate register for pipelining
    Tensor tCrH = thr_mma.make_fragment_A(tCsH(_, _, _, 0));
    Tensor tCrW = thr_mma.make_fragment_B(tCsW(_, _, _, 0));

    // SMEM -> REG tiling
    auto smem_tiled_copy_hidden = typename Traits::TiledLDSM_A{};
    auto smem_thr_copy_hidden = smem_tiled_copy_hidden.get_thread_slice(threadIdx.x);
    Tensor tSsH = smem_thr_copy_hidden.partition_S(sHidden); // (LDSMx4, LDSM_M, LDSM_K, pipe)
    Tensor tCrH_copy_view = smem_thr_copy_hidden.retile_D(tCrH);

    auto smem_tiled_copy_weight = typename Traits::TiledLDSM_B{};
    auto smem_thr_copy_weight = smem_tiled_copy_weight.get_thread_slice(threadIdx.x);
    Tensor tSsW = smem_thr_copy_weight.partition_S(sWeight);
    Tensor tCrW_copy_view = smem_thr_copy_weight.retile_D(tCrW);

    auto smem_tiled_copy_output = make_tiled_copy_C(Copy_Atom<UniversalCopy<uint64_t, uint64_t>, float>{},
                                                    tiled_mma);
    auto smem_thr_copy_output = smem_tiled_copy_output.get_thread_slice(threadIdx.x);
    Tensor tSsO = smem_thr_copy_output.partition_D(sLogit);
    Tensor tCrO_copy_view = smem_thr_copy_output.retile_S(tCrO);

    // LDS.128 or STS.128
    auto gmem_tiled_copy_output = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t, uint128_t>, float>{},
                                                  /*thr=*/Layout<Shape<Int<Traits::threads / 32>, _32>,
                                                                 Stride<_32, _1>>{},
                                                  /*value=*/Layout<Shape<_1, _4>,
                                                                   Stride<_4, _1>>{});
    static_assert(size(gmem_tiled_copy_output) == Traits::threads, 
                  "gmem_tiled_copy_output must have the same number of threads as the number of threads in the block");
    static_assert(Traits::tileN % 128 == 0, "tileN must be divisible by 128");
    auto gmem_thr_copy_output = gmem_tiled_copy_output.get_thread_slice(threadIdx.x);
#if _ENABLE_GMEM_RESULT == 1
    Tensor tCgC_copy_view = gmem_thr_copy_output.partition_D(gC);
#endif
    Tensor tSsO_copy_view = gmem_thr_copy_output.partition_S(sLogit);
    Tensor tCcOutput = gmem_thr_copy_output.partition_D(cC);

    // LDS will reuse the gmem_tiled_copy_output with uint4 for LDS.128
    Tensor tSsLogit_copy_view = gmem_thr_copy_output.partition_S(sLogit);
    Tensor tSrLogit_copy_view = make_fragment_like(tSsO_copy_view);
    Tensor tSrExpLogits = make_fragment_like(tSrLogit_copy_view);

    // (tileM, 1)
    Tensor cLinear = make_identity_tensor(make_shape(size<0>(sLabels), size<1>(sLabels)));
    // using LabelsCopyAtom = Copy_Atom<UniversalCopy<int64_t, int64_t>, int64_t>;
    using LabelsCopyAtom = Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<int64_t>, int64_t>;
    auto smem_tiled_copy_labels = make_tiled_copy(LabelsCopyAtom{},
                                                  /*thr=*/Layout<Shape<Int<Traits::tileM>, Int<Traits::threads / Traits::tileM>>,
                                                                 Stride<_1, Int<Traits::tileM>>>{},
                                                  /*value=*/Layout<Shape<_1>>{});

    static_assert(Traits::threads / Traits::tileM != 0, "threads must be divisible by tileM");
    auto smem_thr_copy_labels = smem_tiled_copy_labels.get_thread_slice(threadIdx.x);
    Tensor tSsLabels = smem_thr_copy_labels.partition_D(sLabels); // (CPY, CPY_M, CPY_N)
    Tensor tSgLabels = smem_thr_copy_labels.partition_S(gLabels); // (CPY, CPY_M, CPY_N)
    Tensor tScLinear = smem_thr_copy_labels.partition_S(cLinear); // (CPY, CPY_M, CPY_N)

    #pragma unroll
    for (int32_t m = 0; m < size<1>(tScLinear); ++m) {
        #pragma unroll
        for (int32_t n = 0; n < size<2>(tScLinear); ++n) {
            if (get<0>(tScLinear(0, m, n)) < (num_tokens - pid_m * Traits::tileM)
                && get<1>(tScLinear(0, m, n)) == 0) {
                copy(smem_tiled_copy_labels,
                    /*src=*/tSgLabels(_, m, n),
                    /*dst=*/tSsLabels(_, m, n));
                
            }
        }
    }
    cp_async_fence();

    float max_val[size<1>(tSrLogit_copy_view)] = {0.f};
    float accumulate[size<1>(tSrLogit_copy_view)] = {0.f};
    float entropy_b[size<1>(tSrLogit_copy_view)] = {0.f};

    int32_t n_tile_count = size<3>(tBgW);
    int32_t vocab_left_bound = pid_n * vocab_per_split;
    int32_t vocab_right_bound = min((pid_n + 1) * vocab_per_split, vocab_size);
    for (int32_t n_tile = 0; n_tile < n_tile_count; ++n_tile) {
        // update the predicate for weight along N
        CUTLASS_PRAGMA_UNROLL
        for (int32_t n = 0; n < size<0>(tBpW); ++n) {
            tBpW(n) = (vocab_left_bound + n_tile * Traits::tileN + get<0>(tBcW(0, n, 0)) < vocab_right_bound);
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

        // Tensor tCsH_p = tCsH(_, _, _, smem_pipe_read); // (MMA, MMA_M, MMA_K)
        Tensor tCsH_p = tSsH(_, _, _, smem_pipe_read);
        // Tensor tCsW_p = tCsW(_, _, _, smem_pipe_read); // (MMA, MMA_N, MMA_K)
        Tensor tCsW_p = tSsW(_, _, _, smem_pipe_read);

        // prefetch first k-block
        constexpr int32_t k_block_max = size<2>(tCrH);
        if constexpr (k_block_max > 1) {
            cp_async_wait<Traits::pipe - 2>();
            __syncthreads();

            // SMEM -> REG
            copy(smem_tiled_copy_hidden, 
                /*src=*/tCsH_p(_, _, Int<0>{}),
                /*dst=*/tCrH_copy_view(_, _, Int<0>{}));
            copy(smem_tiled_copy_weight, 
                /*src=*/tCsW_p(_, _, Int<0>{}),
                /*dst=*/tCrW_copy_view(_, _, Int<0>{}));
        }

        // pipelined main loop on k-tiles
        CUTE_NO_UNROLL
        while (k_tile_count > -(Traits::pipe - 1)) {
            // iterate over k-blocks
            CUTE_UNROLL
            for (int32_t k = 0; k < k_block_max; ++k) {
                // wait for k-tile to be loaded into SMEM
                if (k == k_block_max - 1) {
                    // tCsH_p = tCsH(_, _, _, smem_pipe_read);
                    tCsH_p = tSsH(_, _, _, smem_pipe_read);
                    // tCsW_p = tCsW(_, _, _, smem_pipe_read);
                    tCsW_p = tSsW(_, _, _, smem_pipe_read);
                    cp_async_wait<Traits::pipe - 2>();
                    __syncthreads();
                }

                // SMEM -> REG for k+1
                int32_t k_next = (k + Int<1>{}) % k_block_max;
                // copy(tCsH_p(_,_,k_next), tCrH(_,_,k_next));
                // copy(tCsW_p(_,_,k_next), tCrW(_,_,k_next));
                copy(smem_tiled_copy_hidden,
                     tCsH_p(_, _, k_next), 
                     tCrH_copy_view(_, _, k_next));
                copy(smem_tiled_copy_weight,
                     tCsW_p(_, _, k_next),
                     tCrW_copy_view(_, _, k_next));

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
        // ------- REG -> SMEM
        copy(smem_tiled_copy_output,
                /*src=*/tCrO_copy_view,
                /*dst=*/tSsO);
        __syncthreads();

    #if _ENABLE_GMEM_RESULT == 1
        if (gmem_output_ptr != nullptr) {
            // -------------- REG -> GMEM
            // #pragma unroll
            // for (int32_t k = 0; k < size<0>(tCcC); ++k) {
            //     #pragma unroll
            //     for (int32_t m = 0; m < size<1>(tCcC); ++m) {
            //         #pragma unroll 
            //         for (int32_t n = 0; n < size<2>(tCcC); ++n) {
            //             if (get<0>(tCcC(k, m, n)) < num_tokens - pid_m * Traits::tileM
            //                 && (vocab_left_bound + n_tile * Traits::tileN + get<1>(tCcC(k, m, n)) < vocab_right_bound)) {
            //                 tCgC(k, m, n, n_tile) = tCrO(k, m, n);
            //             }
            //         }
            //     }
            // }

            // SMEM -> GMEM
            #pragma unroll
            for (int32_t m = 0; m < size<1>(tCcOutput); ++m) {
                #pragma unroll
                for (int32_t n = 0; n < size<2>(tCcOutput); ++n) {
                    if (get<0>(tCcOutput(0, m, n)) < (num_tokens - pid_m * Traits::tileM)
                        && (vocab_left_bound + n_tile * Traits::tileN + get<1>(tCcOutput(0, m, n)) < vocab_right_bound)) {
                        copy(gmem_tiled_copy_output,
                            /*src=*/tSsO_copy_view(_, m, n),
                            /*dst=*/tCgC_copy_view(_, m, n, n_tile));
                    }
                }
            }
        }
    #endif

        // ----- SMEM -> LDS
        copy(gmem_tiled_copy_output,
             /*src=*/tSsLogit_copy_view,
             /*dst=*/tSrLogit_copy_view);

        // reduce-max & coefficient
        auto valid = [&](int32_t k, int32_t m, int32_t n, int32_t pid_m) {
            return (get<0>(tCcOutput(k, m, n)) < (num_tokens - pid_m * Traits::tileM)
                    && (vocab_left_bound + n_tile * Traits::tileN + get<1>(tCcOutput(k, m, n)) < vocab_right_bound));
        };
    #if 0 // This branch will use current max_val for exp_logits
        {
            #pragma unroll
            for (int32_t m = 0; m < size<1>(tSrLogit_copy_view); m++) {
                float now_max = 0.f;
                #pragma unroll
                for (int32_t n = 0; n < size<2>(tSrLogit_copy_view); n++) {
                    tSrLogit_copy_view(0, m, n) *= valid(0, m, n, pid_m);
                    tSrLogit_copy_view(1, m, n) *= valid(1, m, n, pid_m);
                    tSrLogit_copy_view(2, m, n) *= valid(2, m, n, pid_m);
                    tSrLogit_copy_view(3, m, n) *= valid(3, m, n, pid_m);

                    float2 max_temp;
                    max_temp.x = fmaxf(tSrLogit_copy_view(0, m, n), tSrLogit_copy_view(1, m, n));
                    max_temp.y = fmaxf(tSrLogit_copy_view(2, m, n), tSrLogit_copy_view(3, m, n));
                    float warp_max = fmaxf(max_temp.x, max_temp.y);
                    warp_max = cg::reduce(warp, warp_max, cg::greater<float>{});
                    now_max = fmaxf(now_max, warp_max);
                }
                // update global max
                float old_max = max_val[m];
                max_val[m] = fmaxf(old_max, now_max);

                float sum_exp_logits = 0.f;
                float sum_exp_logits_mul_logits = 0.f;
                #pragma unroll
                for (int32_t n = 0; n < size<2>(tSrLogit_copy_view); n++) {
                    float4 exp_logits;
                    exp_logits.x = __expf(tSrLogit_copy_view(0, m, n) - max_val[m]);
                    exp_logits.y = __expf(tSrLogit_copy_view(1, m, n) - max_val[m]);
                    exp_logits.z = __expf(tSrLogit_copy_view(2, m, n) - max_val[m]);
                    exp_logits.w = __expf(tSrLogit_copy_view(3, m, n) - max_val[m]);

                    float2 sum_temp;
                    sum_temp.x = __fadd_rn(exp_logits.x, exp_logits.y);
                    sum_temp.y = __fadd_rn(exp_logits.z, exp_logits.w);
                    float warp_sum = __fadd_rn(sum_temp.x, sum_temp.y);
                    warp_sum = cg::reduce(warp, warp_sum, cg::plus<float>{});
                    sum_exp_logits = __fadd_rn(sum_exp_logits, warp_sum);

                    sum_temp.x = __fadd_rn(tSrLogit_copy_view(0, m, n) * exp_logits.x,
                                           tSrLogit_copy_view(1, m, n) * exp_logits.y);
                    sum_temp.y = __fadd_rn(tSrLogit_copy_view(2, m, n) * exp_logits.z,
                                           tSrLogit_copy_view(3, m, n) * exp_logits.w);
                    warp_sum = __fadd_rn(sum_temp.x, sum_temp.y);
                    warp_sum = cg::reduce(warp, warp_sum, cg::plus<float>{});
                    sum_exp_logits_mul_logits = __fadd_rn(sum_exp_logits_mul_logits, warp_sum);

                    // logprobs
                    #pragma unroll
                    for (int32_t k = 0; k < size<0>(tSrLogit_copy_view); k++) {
                        int64_t current_n_idx = rank * vocab_size 
                                                + pid_n * vocab_per_split 
                                                + n_tile * Traits::tileN 
                                                + get<1>(tCcOutput(k, m, n));
                        int64_t m_idx = get<0>(tCcOutput(k, m, n));
                        if (m_idx < (num_tokens - pid_m * Traits::tileM)
                            && sLabels(m_idx, Int<0>{}) == current_n_idx) {
                            gLogprobs(m_idx, Int<0>{}) = tSrLogit_copy_view(k, m, n);
                        }
                    }
                }

                float coefficient = __expf(old_max - max_val[m]);

                accumulate[m] = __fmaf_ieee_rn(coefficient, accumulate[m], sum_exp_logits);
                entropy_b[m] = __fmaf_ieee_rn(entropy_b[m], coefficient, sum_exp_logits_mul_logits);
            }
        }
    #else // this branch will use previous max_val for exp_logits
        {
            #pragma unroll
            for (int32_t m = 0; m < size<1>(tSrLogit_copy_view); m++) {
                float now_max = 0.f;
                float coefficient = 0.f;
                float sum_exp_logits = 0.f;
                float sum_exp_logits_mul_logits = 0.f;
                #pragma unroll
                for (int32_t n = 0; n < size<2>(tSrLogit_copy_view); n++) {
                    tSrLogit_copy_view(0, m, n) *= valid(0, m, n, pid_m);
                    tSrLogit_copy_view(1, m, n) *= valid(1, m, n, pid_m);
                    tSrLogit_copy_view(2, m, n) *= valid(2, m, n, pid_m);
                    tSrLogit_copy_view(3, m, n) *= valid(3, m, n, pid_m);

                    float2 max_temp;
                    max_temp.x = fmaxf(tSrLogit_copy_view(0, m, n), tSrLogit_copy_view(1, m, n));
                    max_temp.y = fmaxf(tSrLogit_copy_view(2, m, n), tSrLogit_copy_view(3, m, n));
                    float warp_max = fmaxf(max_temp.x, max_temp.y);
                    warp_max = cg::reduce(warp, warp_max, cg::greater<float>{});
                    now_max = fmaxf(now_max, warp_max);

                    float4 exp_logits;
                    exp_logits.x = __expf(tSrLogit_copy_view(0, m, n) - max_val[m]);
                    exp_logits.y = __expf(tSrLogit_copy_view(1, m, n) - max_val[m]);
                    exp_logits.z = __expf(tSrLogit_copy_view(2, m, n) - max_val[m]);
                    exp_logits.w = __expf(tSrLogit_copy_view(3, m, n) - max_val[m]);

                    float2 sum_temp;
                    sum_temp.x = __fadd_rn(exp_logits.x, exp_logits.y);
                    sum_temp.y = __fadd_rn(exp_logits.z, exp_logits.w);
                    float warp_sum = __fadd_rn(sum_temp.x, sum_temp.y);
                    warp_sum = cg::reduce(warp, warp_sum, cg::plus<float>{});
                    sum_exp_logits = __fadd_rn(sum_exp_logits, warp_sum);

                    sum_temp.x = __fadd_rn(tSrLogit_copy_view(0, m, n) * exp_logits.x,
                                           tSrLogit_copy_view(1, m, n) * exp_logits.y);
                    sum_temp.y = __fadd_rn(tSrLogit_copy_view(2, m, n) * exp_logits.z,
                                           tSrLogit_copy_view(3, m, n) * exp_logits.w);
                    warp_sum = __fadd_rn(sum_temp.x, sum_temp.y);
                    warp_sum = cg::reduce(warp, warp_sum, cg::plus<float>{});
                    sum_exp_logits_mul_logits = __fadd_rn(sum_exp_logits_mul_logits, warp_sum);

                    // logprobs
                    #pragma unroll
                    for (int32_t k = 0; k < size<0>(tSrLogit_copy_view); k++) {
                        int64_t current_n_idx = rank * vocab_size 
                                                + pid_n * vocab_per_split 
                                                + n_tile * Traits::tileN 
                                                + get<1>(tCcOutput(k, m, n));
                        int64_t m_idx = get<0>(tCcOutput(k, m, n));
                        if (m_idx < (num_tokens - pid_m * Traits::tileM)
                            && sLabels(m_idx, Int<0>{}) == current_n_idx) {
                            gLogprobs(m_idx, Int<0>{}) = tSrLogit_copy_view(k, m, n);
                        }
                    }
                }

                float old_max = max_val[m];
                max_val[m] = fmaxf(old_max, now_max);

                coefficient = __expf(old_max - max_val[m]);

                sum_exp_logits *= coefficient;
                accumulate[m] = __fmaf_ieee_rn(coefficient, accumulate[m], sum_exp_logits);

                sum_exp_logits_mul_logits *= coefficient;
                entropy_b[m] = __fmaf_ieee_rn(entropy_b[m], coefficient, sum_exp_logits_mul_logits);
            }
        }
    #endif
    } // n_tile
    {
        {
            if (warp.thread_rank() == 0) {
                #pragma unroll
                for (int32_t m = 0; m < size<1>(tSrLogit_copy_view); m++) {
                    int32_t m_idx = get<0>(tCcOutput(0, m, 0));
                    if (m_idx < (num_tokens - pid_m * Traits::tileM)) {
                        gMax(m_idx, Int<0>{}) = max_val[m];
                        gAcc(m_idx, Int<0>{}) = accumulate[m];
                        gEntropyB(m_idx, Int<0>{}) = entropy_b[m];
                    }
                }
            }
        }
    }
}

#undef _ENABLE_GMEM_RESULT

} // namespace lce
#endif


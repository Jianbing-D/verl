#include "interfaces.h"
#include "sm80.h"

namespace lce {

template <>
void forward_mainloop<float, float>(int32_t rank,
                                    void *hidden_ptr,
                                    void *weight_ptr,
                                    int64_t *labels_ptr,
                                    int32_t num_tokens,
                                    int32_t vocab_size,
                                    int32_t vocab_per_split,
                                    void *max_ptr,
                                    void *acc_ptr,
                                    void *entropy_b_ptr,
                                    void *logprobs_ptr,
                                    float *gmem_output_ptr,
                                    cudaStream_t stream) {}

template <>
void forward_mainloop<__nv_bfloat16, __nv_bfloat16>(int32_t rank,
                                                    void *hidden_ptr,
                                                    void *weight_ptr,
                                                    int64_t *labels_ptr,
                                                    int32_t num_tokens,
                                                    int32_t vocab_size,
                                                    int32_t vocab_per_split,
                                                    void *max_ptr,
                                                    void *acc_ptr,
                                                    void *entropy_b_ptr,
                                                    void *logprobs_ptr,
                                                    float *gmem_output_ptr,
                                                    cudaStream_t stream) {
    // first, lets check whether the GEMM is correct
    using Traits = lce::Traits<__nv_bfloat16, __nv_bfloat16, 4096>;

    int32_t num_splits = (vocab_size + vocab_per_split - 1) / vocab_per_split;
    int32_t last_split_size = vocab_size - (num_splits - 1) * vocab_per_split;
    if (last_split_size % 4 != 0) {
        // NOTE: such requirement is due to the GMEM vectorized store
        throw std::invalid_argument("last split size must be divisible by 4 for address alignment");
    }

    // thread-block swizzle
    int32_t num_blocks = (num_tokens + Traits::tileM - 1) / Traits::tileM;
    num_blocks *= Traits::threadBlockSwizzleSize;

    num_blocks *= ((num_splits + Traits::threadBlockSwizzleSize - 1) / Traits::threadBlockSwizzleSize);

    dim3 block(Traits::threads, 1, 1);
    dim3 grid(num_blocks, 1, 1);

    auto kernel = lce::forward_mainloop_kernel<Traits>;
    if (Traits::smem_bytes >= 48 * 1024ul) {
        CUDA_THROW(cudaFuncSetAttribute(kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             Traits::smem_bytes));
    }

#if 0
    // print_latex(Traits::SmemLayoutAtom{});
    print_latex(Traits::SmemLayoutHidden{});
    printf("block: (%d, %d, %d), grid: (%d, %d, %d), smem_bytes: %ld\n",
        block.x, block.y, block.z, grid.x, grid.y, grid.z, Traits::smem_bytes);
#endif

    kernel<<<grid, block, Traits::smem_bytes, stream>>>(
        rank,
        reinterpret_cast<typename Traits::IN_DTYPE*>(hidden_ptr),
        reinterpret_cast<typename Traits::IN_DTYPE*>(weight_ptr),
        labels_ptr,
        num_tokens,
        vocab_size,
        vocab_per_split,
        num_splits,
        reinterpret_cast<float*>(max_ptr),
        reinterpret_cast<float*>(acc_ptr),
        reinterpret_cast<float*>(entropy_b_ptr),
        reinterpret_cast<float*>(logprobs_ptr),
        reinterpret_cast<float*>(gmem_output_ptr)
    );
    CUDA_THROW(cudaGetLastError());
}

} // namespace lce
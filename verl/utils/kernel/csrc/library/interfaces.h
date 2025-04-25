#ifndef LCE_INTERFACES_H
#define LCE_INTERFACES_H

#include <cstdint>

namespace lce {

#define CUDA_NOTHROW(cmd)                                                                   \
    do {                                                                                    \
        cudaError_t err = (cmd);                                                            \
        if (err != cudaSuccess) {                                                           \
            printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));  \
            exit(1);                                                                        \
        }                                                                                   \
    } while (0)

#define CUDA_THROW(cmd)                                                                          \
    do {                                                                                         \
        cudaError_t err = (cmd);                                                                 \
        if (err != cudaSuccess) {                                                                \
            throw std::runtime_error(std::string("CUDA error: ") + __FILE__ + ":" +              \
                                     std::to_string(__LINE__) + ": " + cudaGetErrorString(err) + \
                                     std::string(" in ") + #cmd);                                \
        }                                                                                        \
    } while (0)


template <typename InT, typename OutT>
void forward_mainloop(int32_t rank,
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
                      cudaStream_t stream);

template <typename InT, typename OutT>
void backward_d_logits(int32_t num_tokens,
                       int32_t hidden_size,
                       int32_t vocab_size,
                       int32_t rank,
                       InT *hidden_ptr,
                       InT *weight_ptr,
                       int64_t *labels_ptr,
                       float *maximum_ptr,
                       float *accumulate_ptr,
                       float *entropy_b_ptr,
                       float *grad_entropy_ptr,
                       float *grad_logprobs_ptr,
                       OutT *grad_logits_ptr,
                       float *gmem_output_ptr,
                       cudaStream_t stream);


} // namespace lce
#endif

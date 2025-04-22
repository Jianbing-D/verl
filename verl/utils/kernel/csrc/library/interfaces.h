#ifndef LCE_INTERFACES_H
#define LCE_INTERFACES_H

#include <cstdint>

namespace lce {

template <typename InT, typename OutT>
void forward_mainloop(int32_t rank,
                      void *hidden_ptr,
                      int32_t stride_hidden_m, int32_t stride_hidden_k,
                      void *weight_ptr,
                      int32_t stride_weight_n, int32_t stride_weight_k,
                      uint64_t *labels_ptr,
                      int32_t num_tokens,
                      int32_t vocab_size,
                      int32_t vocab_per_split,
                      float *gmem_output_ptr,
                      cudaStream_t stream);

} // namespace lce
#endif

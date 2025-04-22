#include <torch/torch.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <string>

#include "interfaces.h"


#define KERNEL_WITH_DTYPE(dtype, DTYPE, ...)                    \
    if (torch::kFloat32 == dtype) {                             \
        using DTYPE = float;                                    \
        { __VA_ARGS__ }                                         \
    } else if (torch::kBFloat16 == dtype) {                     \
        using DTYPE = __nv_bfloat16;                            \
        { __VA_ARGS__ }                                         \
    } else {                                                    \
        TORCH_CHECK(false, "Unsupported dtype: ", dtype);       \
    }   

void forward_mainloop(torch::Tensor &hidden, int32_t stride_hidden_m, int32_t stride_hidden_k,
                      torch::Tensor &weight, int32_t stride_weight_n, int32_t stride_weight_k,
                      torch::Tensor &labels, int32_t labels_stride,
                      int32_t rank, int32_t num_tokens, int32_t vocab_size, int32_t hidden_size,
                      int32_t vocab_per_split,
                      torch::Tensor &_max, int32_t stride_max_m, int32_t stride_max_n,
                      torch::Tensor &_acc, int32_t stride_acc_m, int32_t stride_acc_n,
                      torch::Tensor &_entropy_b, int32_t stride_entropy_b_m, int32_t stride_entropy_b_n,
                      torch::Tensor &final_logprobs,
                      torch::Tensor &final_logprobs_scalar,
                      std::optional<torch::Tensor> gmem_output = std::nullopt) {
    auto stream = c10::cuda::getCurrentCUDAStream(hidden.device().index());

    KERNEL_WITH_DTYPE(hidden.dtype(), IN_DTYPE,
        lce::forward_mainloop<IN_DTYPE, IN_DTYPE>(
            rank, 
            hidden.data_ptr(), stride_hidden_m, stride_hidden_k,
            weight.data_ptr(), stride_weight_n, stride_weight_k,
            reinterpret_cast<uint64_t *>(labels.data_ptr()),
            num_tokens,
            vocab_size,
            vocab_per_split,
            gmem_output.has_value() ? (float*)gmem_output.value().data_ptr() : (float*)nullptr,
            stream
        );
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_mainloop", &forward_mainloop, "Linear Cross Entropy Forward Mainloop");
}
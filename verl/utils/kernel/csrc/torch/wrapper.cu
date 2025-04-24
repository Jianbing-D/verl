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

    TORCH_CHECK(hidden.dim() == 2, "hidden must be a 2D tensor");
    TORCH_CHECK(weight.dim() == 2, "weight must be a 2D tensor");
    TORCH_CHECK(hidden.size(0) == num_tokens, "hidden.size(0) must be equal to num_tokens");
    TORCH_CHECK(hidden.size(1) == hidden_size, "hidden.size(1) must be equal to hidden_size");
    TORCH_CHECK(weight.size(0) == vocab_size, "weight.size(0) must be equal to vocab_size");
    TORCH_CHECK(weight.size(1) == hidden_size, "weight.size(1) must be equal to hidden_size");
    TORCH_CHECK(labels.dim() == 1, "labels must be a 1D tensor");
    TORCH_CHECK(_max.dim() == 2, "_max must be a 2D tensor");
    TORCH_CHECK(_acc.dim() == 2, "_acc must be a 2D tensor");
    TORCH_CHECK(_entropy_b.dim() == 2, "_entropy_b must be a 2D tensor");
    TORCH_CHECK(final_logprobs.dim() == 1, "final_logprobs must be a 1D tensor");

    TORCH_CHECK(hidden.dtype() == torch::kBFloat16, "hidden must be a bfloat16 tensor");
    TORCH_CHECK(weight.dtype() == torch::kBFloat16, "weight must be a bfloat16 tensor");
    TORCH_CHECK(labels.dtype() == torch::kInt64, "labels must be a int64 tensor");
    TORCH_CHECK(_max.dtype() == torch::kFloat32, "_max must be a float32 tensor");
    TORCH_CHECK(_acc.dtype() == torch::kFloat32, "_acc must be a float32 tensor");
    TORCH_CHECK(_entropy_b.dtype() == torch::kFloat32, "_entropy_b must be a float32 tensor");
    TORCH_CHECK(final_logprobs.dtype() == torch::kFloat32, "final_logprobs must be a float32 tensor");

    TORCH_CHECK(hidden.is_cuda(), "hidden must be a cuda tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a cuda tensor");
    TORCH_CHECK(labels.is_cuda(), "labels must be a cuda tensor");
    TORCH_CHECK(_max.is_cuda(), "_max must be a cuda tensor");
    TORCH_CHECK(_acc.is_cuda(), "_acc must be a cuda tensor");
    TORCH_CHECK(_entropy_b.is_cuda(), "_entropy_b must be a cuda tensor");
    TORCH_CHECK(final_logprobs.is_cuda(), "final_logprobs must be a cuda tensor");
    
    TORCH_CHECK(hidden.is_contiguous(), "hidden must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(labels.is_contiguous(), "labels must be contiguous");
    TORCH_CHECK(_max.is_contiguous(), "_max must be contiguous");
    TORCH_CHECK(_acc.is_contiguous(), "_acc must be contiguous");
    TORCH_CHECK(_entropy_b.is_contiguous(), "_entropy_b must be contiguous");
    TORCH_CHECK(final_logprobs.is_contiguous(), "final_logprobs must be contiguous");

    KERNEL_WITH_DTYPE(hidden.dtype(), IN_DTYPE,
        lce::forward_mainloop<IN_DTYPE, IN_DTYPE>(
            rank, 
            hidden.data_ptr(),
            weight.data_ptr(),
            reinterpret_cast<int64_t *>(labels.data_ptr()),
            num_tokens,
            vocab_size,
            vocab_per_split,
            _max.data_ptr(), 
            _acc.data_ptr(),
            _entropy_b.data_ptr(),
            final_logprobs.data_ptr(),
            gmem_output.has_value() ? (float*)gmem_output.value().data_ptr() : (float*)nullptr,
            stream
        );
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_mainloop", &forward_mainloop, "Linear Cross Entropy Forward Mainloop");
}
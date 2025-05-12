
#include <torch/torch.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDABlas.h>
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

void backward_d_logits(int32_t num_tokens, int32_t hidden_size, int32_t vocab_size, int32_t rank,
                       torch::Tensor &hidden, int32_t stride_hidden_m, int32_t stride_hidden_k,
                       torch::Tensor &weight, int32_t stride_weight_n, int32_t stride_weight_k,
                       torch::Tensor &labels, int32_t labels_stride,
                       torch::Tensor &maximum, int32_t stride_maximum,
                       torch::Tensor &accumulate, int32_t stride_accumulate,
                       torch::Tensor &entropy_b, int32_t stride_entropy_b,
                       torch::Tensor &grad_entropy, int32_t stride_grad_entropy,
                       torch::Tensor &grad_logprobs, int32_t stride_grad_logprobs,
                       torch::Tensor &grad_logits, int32_t stride_grad_logits_m, int32_t stride_grad_logits_n,
                       std::optional<torch::Tensor> gmem_output = std::nullopt) {
    auto stream = c10::cuda::getCurrentCUDAStream(hidden.device().index());

    TORCH_CHECK(hidden.dim() == 2, "hidden must be a 2D tensor");
    TORCH_CHECK(weight.dim() == 2, "weight must be a 2D tensor");
    TORCH_CHECK(hidden.size(0) == num_tokens, "hidden.size(0) must be equal to num_tokens");
    TORCH_CHECK(hidden.size(1) == hidden_size, "hidden.size(1) must be equal to hidden_size");
    TORCH_CHECK(weight.size(0) == vocab_size, "weight.size(0) must be equal to vocab_size");
    TORCH_CHECK(weight.size(1) == hidden_size, "weight.size(1) must be equal to hidden_size");
    TORCH_CHECK(labels.dim() == 1, "labels must be a 1D tensor");
    TORCH_CHECK(labels.size(0) == num_tokens, "labels.size(0) must be equal to num_tokens");
    TORCH_CHECK(maximum.dim() == 1, "maximum must be a 1D tensor");
    TORCH_CHECK(maximum.size(0) == num_tokens, "maximum.size(0) must be equal to num_tokens");
    TORCH_CHECK(accumulate.dim() == 1, "accumulate must be a 1D tensor");
    TORCH_CHECK(accumulate.size(0) == num_tokens, "accumulate.size(0) must be equal to num_tokens");
    TORCH_CHECK(entropy_b.dim() == 1, "entropy_b must be a 1D tensor");
    TORCH_CHECK(entropy_b.size(0) == num_tokens, "entropy_b.size(0) must be equal to num_tokens");
    TORCH_CHECK(grad_entropy.dim() == 1, "grad_entropy must be a 1D tensor");
    TORCH_CHECK(grad_entropy.size(0) == num_tokens, "grad_entropy.size(0) must be equal to num_tokens");
    TORCH_CHECK(grad_logprobs.dim() == 1, "grad_logprobs must be a 1D tensor");
    TORCH_CHECK(grad_logprobs.size(0) == num_tokens, "grad_logprobs.size(0) must be equal to num_tokens");
    TORCH_CHECK(grad_logits.dim() == 2, "grad_logits must be a 2D tensor");
    TORCH_CHECK(grad_logits.size(0) == num_tokens, "grad_logits.size(0) must be equal to num_tokens");
    TORCH_CHECK(grad_logits.size(1) == vocab_size, "grad_logits.size(1) must be equal to vocab_size");

    TORCH_CHECK(hidden.dtype() == torch::kBFloat16, "hidden must be a bfloat16 tensor");
    TORCH_CHECK(weight.dtype() == torch::kBFloat16, "weight must be a bfloat16 tensor");
    TORCH_CHECK(labels.dtype() == torch::kInt64, "labels must be a int64 tensor");
    TORCH_CHECK(maximum.dtype() == torch::kFloat32, "maximum must be a float32 tensor");
    TORCH_CHECK(accumulate.dtype() == torch::kFloat32, "accumulate must be a float32 tensor");
    TORCH_CHECK(entropy_b.dtype() == torch::kFloat32, "entropy_b must be a float32 tensor");
    TORCH_CHECK(grad_entropy.dtype() == torch::kFloat32, "grad_entropy must be a float32 tensor");
    TORCH_CHECK(grad_logprobs.dtype() == torch::kFloat32, "grad_logprobs must be a float32 tensor");
    TORCH_CHECK(grad_logits.dtype() == torch::kBFloat16, "grad_logits must be a bfloat16 tensor");

    TORCH_CHECK(hidden.is_cuda(), "hidden must be a cuda tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a cuda tensor");
    TORCH_CHECK(labels.is_cuda(), "labels must be a cuda tensor");
    TORCH_CHECK(maximum.is_cuda(), "maximum must be a cuda tensor");
    TORCH_CHECK(accumulate.is_cuda(), "accumulate must be a cuda tensor");
    TORCH_CHECK(entropy_b.is_cuda(), "entropy_b must be a cuda tensor");
    TORCH_CHECK(grad_entropy.is_cuda(), "grad_entropy must be a cuda tensor");
    TORCH_CHECK(grad_logprobs.is_cuda(), "grad_logprobs must be a cuda tensor");
    TORCH_CHECK(grad_logits.is_cuda(), "grad_logits must be a cuda tensor");

    TORCH_CHECK(hidden.is_contiguous(), "hidden must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(labels.is_contiguous(), "labels must be contiguous");
    TORCH_CHECK(maximum.is_contiguous(), "maximum must be contiguous");
    TORCH_CHECK(accumulate.is_contiguous(), "accumulate must be contiguous");
    TORCH_CHECK(entropy_b.is_contiguous(), "entropy_b must be contiguous");
    TORCH_CHECK(grad_entropy.is_contiguous(), "grad_entropy must be contiguous");
    TORCH_CHECK(grad_logprobs.is_contiguous(), "grad_logprobs must be contiguous");
    TORCH_CHECK(grad_logits.is_contiguous(), "grad_logits must be contiguous");
    
    KERNEL_WITH_DTYPE(hidden.dtype(), IN_DTYPE,
        using OUT_DTYPE = IN_DTYPE;
        lce::backward_d_logits<IN_DTYPE, OUT_DTYPE>(
            num_tokens, hidden_size, vocab_size, rank,
            reinterpret_cast<IN_DTYPE *>(hidden.data_ptr()),
            reinterpret_cast<IN_DTYPE *>(weight.data_ptr()),
            reinterpret_cast<int64_t *>(labels.data_ptr()),
            reinterpret_cast<float *>(maximum.data_ptr()),
            reinterpret_cast<float *>(accumulate.data_ptr()),
            reinterpret_cast<float *>(entropy_b.data_ptr()),
            reinterpret_cast<float *>(grad_entropy.data_ptr()),
            reinterpret_cast<float *>(grad_logprobs.data_ptr()),
            reinterpret_cast<OUT_DTYPE *>(grad_logits.data_ptr()),
            gmem_output.has_value() ? (float*)gmem_output.value().data_ptr() : (float*)nullptr,
            stream
        );
    );
}

void backward_d_logits_and_cublas_matmul(int32_t num_tokens, int32_t hidden_size, int32_t vocab_size, int32_t rank,
                                        torch::Tensor &hidden, int32_t stride_hidden_m, int32_t stride_hidden_k,
                                        torch::Tensor &weight, int32_t stride_weight_n, int32_t stride_weight_k,
                                        torch::Tensor &labels, int32_t labels_stride,
                                        torch::Tensor &maximum, int32_t stride_maximum,
                                        torch::Tensor &accumulate, int32_t stride_accumulate,
                                        torch::Tensor &entropy_b, int32_t stride_entropy_b,
                                        torch::Tensor &grad_entropy, int32_t stride_grad_entropy,
                                        torch::Tensor &grad_logprobs, int32_t stride_grad_logprobs,
                                        torch::Tensor &grad_logits, int32_t stride_grad_logits_m, int32_t stride_grad_logits_n,
                                        torch::Tensor &grad_hidden, int32_t stride_grad_hidden_m, int32_t stride_grad_hidden_k,
                                        torch::Tensor &grad_weight, int32_t stride_grad_weight_n, int32_t stride_grad_weight_k,
                                        std::optional<torch::Tensor> gmem_output = std::nullopt) {
    auto stream = c10::cuda::getCurrentCUDAStream(hidden.device().index());
    // Get the current cuBLAS handle from PyTorch
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();

    TORCH_CHECK(hidden.dim() == 2, "hidden must be a 2D tensor");
    TORCH_CHECK(weight.dim() == 2, "weight must be a 2D tensor");
    TORCH_CHECK(hidden.size(0) == num_tokens, "hidden.size(0) must be equal to num_tokens");
    TORCH_CHECK(hidden.size(1) == hidden_size, "hidden.size(1) must be equal to hidden_size");
    TORCH_CHECK(weight.size(0) == vocab_size, "weight.size(0) must be equal to vocab_size");
    TORCH_CHECK(weight.size(1) == hidden_size, "weight.size(1) must be equal to hidden_size");
    TORCH_CHECK(labels.dim() == 1, "labels must be a 1D tensor");
    TORCH_CHECK(labels.size(0) == num_tokens, "labels.size(0) must be equal to num_tokens");
    TORCH_CHECK(maximum.dim() == 1, "maximum must be a 1D tensor");
    TORCH_CHECK(maximum.size(0) == num_tokens, "maximum.size(0) must be equal to num_tokens");
    TORCH_CHECK(accumulate.dim() == 1, "accumulate must be a 1D tensor");
    TORCH_CHECK(accumulate.size(0) == num_tokens, "accumulate.size(0) must be equal to num_tokens");
    TORCH_CHECK(entropy_b.dim() == 1, "entropy_b must be a 1D tensor");
    TORCH_CHECK(entropy_b.size(0) == num_tokens, "entropy_b.size(0) must be equal to num_tokens");
    TORCH_CHECK(grad_entropy.dim() == 1, "grad_entropy must be a 1D tensor");
    TORCH_CHECK(grad_entropy.size(0) == num_tokens, "grad_entropy.size(0) must be equal to num_tokens");
    TORCH_CHECK(grad_logprobs.dim() == 1, "grad_logprobs must be a 1D tensor");
    TORCH_CHECK(grad_logprobs.size(0) == num_tokens, "grad_logprobs.size(0) must be equal to num_tokens");
    TORCH_CHECK(grad_logits.dim() == 2, "grad_logits must be a 2D tensor");
    TORCH_CHECK(grad_logits.size(0) == num_tokens, "grad_logits.size(0) must be equal to num_tokens");
    TORCH_CHECK(grad_logits.size(1) == vocab_size, "grad_logits.size(1) must be equal to vocab_size");
    TORCH_CHECK(grad_hidden.dim() == 2, "grad_hidden must be a 2D tensor");
    TORCH_CHECK(grad_hidden.size(0) == num_tokens, "grad_hidden.size(0) must be equal to num_tokens");
    TORCH_CHECK(grad_hidden.size(1) == hidden_size, "grad_hidden.size(1) must be equal to hidden_size");
    TORCH_CHECK(grad_weight.dim() == 2, "grad_weight must be a 2D tensor");
    TORCH_CHECK(grad_weight.size(0) == vocab_size, "grad_weight.size(0) must be equal to vocab_size");
    TORCH_CHECK(grad_weight.size(1) == hidden_size, "grad_weight.size(1) must be equal to hidden_size");

    TORCH_CHECK(hidden.dtype() == torch::kBFloat16, "hidden must be a bfloat16 tensor");
    TORCH_CHECK(weight.dtype() == torch::kBFloat16, "weight must be a bfloat16 tensor");
    TORCH_CHECK(labels.dtype() == torch::kInt64, "labels must be a int64 tensor");
    TORCH_CHECK(maximum.dtype() == torch::kFloat32, "maximum must be a float32 tensor");
    TORCH_CHECK(accumulate.dtype() == torch::kFloat32, "accumulate must be a float32 tensor");
    TORCH_CHECK(entropy_b.dtype() == torch::kFloat32, "entropy_b must be a float32 tensor");
    TORCH_CHECK(grad_entropy.dtype() == torch::kFloat32, "grad_entropy must be a float32 tensor");
    TORCH_CHECK(grad_logprobs.dtype() == torch::kFloat32, "grad_logprobs must be a float32 tensor");
    TORCH_CHECK(grad_logits.dtype() == torch::kBFloat16, "grad_logits must be a bfloat16 tensor");
    TORCH_CHECK(grad_hidden.dtype() == torch::kBFloat16, "grad_hidden must be a bfloat16 tensor");
    TORCH_CHECK(grad_weight.dtype() == torch::kBFloat16, "grad_weight must be a bfloat16 tensor");

    TORCH_CHECK(hidden.is_cuda(), "hidden must be a cuda tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a cuda tensor");
    TORCH_CHECK(labels.is_cuda(), "labels must be a cuda tensor");
    TORCH_CHECK(maximum.is_cuda(), "maximum must be a cuda tensor");
    TORCH_CHECK(accumulate.is_cuda(), "accumulate must be a cuda tensor");
    TORCH_CHECK(entropy_b.is_cuda(), "entropy_b must be a cuda tensor");
    TORCH_CHECK(grad_entropy.is_cuda(), "grad_entropy must be a cuda tensor");
    TORCH_CHECK(grad_logprobs.is_cuda(), "grad_logprobs must be a cuda tensor");
    TORCH_CHECK(grad_logits.is_cuda(), "grad_logits must be a cuda tensor");
    TORCH_CHECK(grad_hidden.is_cuda(), "grad_hidden must be a cuda tensor");
    TORCH_CHECK(grad_weight.is_cuda(), "grad_weight must be a cuda tensor");

    TORCH_CHECK(hidden.is_contiguous(), "hidden must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(labels.is_contiguous(), "labels must be contiguous");
    TORCH_CHECK(maximum.is_contiguous(), "maximum must be contiguous");
    TORCH_CHECK(accumulate.is_contiguous(), "accumulate must be contiguous");
    TORCH_CHECK(entropy_b.is_contiguous(), "entropy_b must be contiguous");
    TORCH_CHECK(grad_entropy.is_contiguous(), "grad_entropy must be contiguous");
    TORCH_CHECK(grad_logprobs.is_contiguous(), "grad_logprobs must be contiguous");
    TORCH_CHECK(grad_logits.is_contiguous(), "grad_logits must be contiguous");
    TORCH_CHECK(grad_hidden.is_contiguous(), "grad_hidden must be contiguous");
    TORCH_CHECK(grad_weight.is_contiguous(), "grad_weight must be contiguous");
    
    KERNEL_WITH_DTYPE(hidden.dtype(), IN_DTYPE,
        using OUT_DTYPE = IN_DTYPE;
        lce::backward_d_logits<IN_DTYPE, OUT_DTYPE>(
            num_tokens, hidden_size, vocab_size, rank,
            reinterpret_cast<IN_DTYPE *>(hidden.data_ptr()),
            reinterpret_cast<IN_DTYPE *>(weight.data_ptr()),
            reinterpret_cast<int64_t *>(labels.data_ptr()),
            reinterpret_cast<float *>(maximum.data_ptr()),
            reinterpret_cast<float *>(accumulate.data_ptr()),
            reinterpret_cast<float *>(entropy_b.data_ptr()),
            reinterpret_cast<float *>(grad_entropy.data_ptr()),
            reinterpret_cast<float *>(grad_logprobs.data_ptr()),
            reinterpret_cast<OUT_DTYPE *>(grad_logits.data_ptr()),
            gmem_output.has_value() ? (float*)gmem_output.value().data_ptr() : (float*)nullptr,
            stream
        );
    );

    float alpha = 1.0f;
    float beta = 0.0f;
    CUBLAS_THROW(cublasSetStream(handle, stream));
    CUBLAS_THROW(cublasGemmEx(handle,
                            CUBLAS_OP_N,
                            CUBLAS_OP_N,
                            hidden_size,
                            num_tokens,
                            vocab_size,
                            &alpha,
                            weight.data_ptr(),
                            CUDA_R_16BF,
                            hidden_size,
                            grad_logits.data_ptr(),
                            CUDA_R_16BF,
                            vocab_size,
                            &beta,
                            grad_hidden.data_ptr(),
                            CUDA_R_16BF,
                            hidden_size,
                            CUBLAS_COMPUTE_32F,
                            CUBLAS_GEMM_DEFAULT));
    CUBLAS_THROW(cublasGemmEx(handle,
                        CUBLAS_OP_N,        // Changed to N for hidden
                        CUBLAS_OP_T,        // Changed to T for grad_logits
                        hidden_size,        // m - output cols
                        vocab_size,         // n - output rows
                        num_tokens,         // k - inner dimension
                        &alpha,
                        hidden.data_ptr(),  // hidden matrix (not transposed)
                        CUDA_R_16BF,
                        hidden_size,        // leading dimension of hidden
                        grad_logits.data_ptr(), // grad_logits matrix (transposed)
                        CUDA_R_16BF,
                        vocab_size,         // leading dimension of grad_logits
                        &beta,
                        grad_weight.data_ptr(), // output: grad_weight
                        CUDA_R_16BF,
                        hidden_size,        // leading dimension of grad_weight
                        CUBLAS_COMPUTE_32F,
                        CUBLAS_GEMM_DEFAULT));
}

void cublas_matmul_after_d_logits(int32_t num_tokens, int32_t hidden_size, int32_t vocab_size, int32_t rank,
                                torch::Tensor &hidden, int32_t stride_hidden_m, int32_t stride_hidden_k,
                                torch::Tensor &weight, int32_t stride_weight_n, int32_t stride_weight_k,
                                torch::Tensor &grad_logits, int32_t stride_grad_logits_m, int32_t stride_grad_logits_n,
                                torch::Tensor &grad_hidden, int32_t stride_grad_hidden_m, int32_t stride_grad_hidden_k,
                                torch::Tensor &grad_weight, int32_t stride_grad_weight_n, int32_t stride_grad_weight_k) {
    auto stream = c10::cuda::getCurrentCUDAStream(hidden.device().index());
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();

    TORCH_CHECK(hidden.dim() == 2, "hidden must be a 2D tensor");
    TORCH_CHECK(weight.dim() == 2, "weight must be a 2D tensor");
    TORCH_CHECK(hidden.size(0) == num_tokens, "hidden.size(0) must be equal to num_tokens");
    TORCH_CHECK(hidden.size(1) == hidden_size, "hidden.size(1) must be equal to hidden_size");
    TORCH_CHECK(weight.size(0) == vocab_size, "weight.size(0) must be equal to vocab_size");
    TORCH_CHECK(weight.size(1) == hidden_size, "weight.size(1) must be equal to hidden_size");
    TORCH_CHECK(grad_logits.dim() == 2, "grad_logits must be a 2D tensor");
    TORCH_CHECK(grad_logits.size(0) == num_tokens, "grad_logits.size(0) must be equal to num_tokens");
    TORCH_CHECK(grad_logits.size(1) == vocab_size, "grad_logits.size(1) must be equal to vocab_size");
    TORCH_CHECK(grad_hidden.dim() == 2, "grad_hidden must be a 2D tensor");
    TORCH_CHECK(grad_hidden.size(0) == num_tokens, "grad_hidden.size(0) must be equal to num_tokens");
    TORCH_CHECK(grad_hidden.size(1) == hidden_size, "grad_hidden.size(1) must be equal to hidden_size");
    TORCH_CHECK(grad_weight.dim() == 2, "grad_weight must be a 2D tensor");
    TORCH_CHECK(grad_weight.size(0) == vocab_size, "grad_weight.size(0) must be equal to vocab_size");
    TORCH_CHECK(grad_weight.size(1) == hidden_size, "grad_weight.size(1) must be equal to hidden_size");

    TORCH_CHECK(hidden.dtype() == torch::kBFloat16, "hidden must be a bfloat16 tensor");
    TORCH_CHECK(weight.dtype() == torch::kBFloat16, "weight must be a bfloat16 tensor");
    TORCH_CHECK(grad_logits.dtype() == torch::kBFloat16, "grad_logits must be a bfloat16 tensor");
    TORCH_CHECK(grad_hidden.dtype() == torch::kBFloat16, "grad_hidden must be a bfloat16 tensor");
    TORCH_CHECK(grad_weight.dtype() == torch::kBFloat16, "grad_weight must be a bfloat16 tensor");

    TORCH_CHECK(hidden.is_cuda(), "hidden must be a cuda tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a cuda tensor");
    TORCH_CHECK(grad_logits.is_cuda(), "grad_logits must be a cuda tensor");
    TORCH_CHECK(grad_hidden.is_cuda(), "grad_hidden must be a cuda tensor");
    TORCH_CHECK(grad_weight.is_cuda(), "grad_weight must be a cuda tensor");

    TORCH_CHECK(hidden.is_contiguous(), "hidden must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(grad_logits.is_contiguous(), "grad_logits must be contiguous");
    TORCH_CHECK(grad_hidden.is_contiguous(), "grad_hidden must be contiguous");
    TORCH_CHECK(grad_weight.is_contiguous(), "grad_weight must be contiguous");
    
    float alpha = 1.0f;
    float beta = 0.0f;
    CUBLAS_THROW(cublasSetStream(handle, stream));
    CUBLAS_THROW(cublasGemmEx(handle,
                            CUBLAS_OP_N,
                            CUBLAS_OP_N,
                            hidden_size,
                            num_tokens,
                            vocab_size,
                            &alpha,
                            weight.data_ptr(),
                            CUDA_R_16BF,
                            hidden_size,
                            grad_logits.data_ptr(),
                            CUDA_R_16BF,
                            vocab_size,
                            &beta,
                            grad_hidden.data_ptr(),
                            CUDA_R_16BF,
                            hidden_size,
                            CUBLAS_COMPUTE_32F,
                            CUBLAS_GEMM_DEFAULT));
    // CUBLAS_THROW(cublasGemmEx(handle,
    //                             CUBLAS_OP_N,        
    //                             CUBLAS_OP_T,      
    //                             hidden_size,    
    //                             vocab_size,      
    //                             num_tokens,        
    //                             &alpha,
    //                             hidden.data_ptr(),  
    //                             CUDA_R_16BF,
    //                             hidden_size,        
    //                             grad_logits.data_ptr(), 
    //                             CUDA_R_16BF,
    //                             vocab_size,         
    //                             &beta,
    //                             grad_weight.data_ptr(), 
    //                             CUDA_R_16BF,
    //                             hidden_size,        
    //                             CUBLAS_COMPUTE_32F,
    //                             CUBLAS_GEMM_DEFAULT));

    CUBLAS_THROW(cublasSgemmEx(handle,
                               CUBLAS_OP_N,
                               CUBLAS_OP_T,
                               hidden_size,
                               vocab_size,
                               num_tokens,
                               &alpha,
                               hidden.data_ptr(),
                               CUDA_R_16BF,
                               hidden_size,
                               grad_logits.data_ptr(),
                               CUDA_R_16BF,
                               vocab_size,
                               &beta,
                               grad_weight.data_ptr(),
                               CUDA_R_16BF,
                               hidden_size));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_mainloop", &forward_mainloop, "Linear Cross Entropy Forward Mainloop");
    m.def("backward_d_logits", &backward_d_logits, "Linear Cross Entropy Backward d_logits");
    m.def("backward_d_logits_and_cublas_matmul", 
          &backward_d_logits_and_cublas_matmul, 
          "Linear Cross Entropy Backward d_logits following cublas matmul");
    m.def("cublas_matmul_after_d_logits",
          &cublas_matmul_after_d_logits,
          "cublas matmul after d_logits");
}

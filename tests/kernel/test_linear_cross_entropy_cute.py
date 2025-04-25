import torch


# FIXME: remove these manually included paths
import os
import sys
extension_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../verl/utils/kernel/csrc/torch/"))

sys.path.append(extension_path)
import linear_cross_entropy_extension as lce_ext

# ncu --set full --nvtx --nvtx-include "forward_mainloop/" -f -o cute python tests/kernel/test_linear_cross_entropy_cute.py
# ncu --set full -o cute python tests/kernel/test_linear_cross_entropy_cute.py


class TestLinearCrossEntropyCUTE:
    def cleanup(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        import gc
        gc.collect()
        torch.cuda.synchronize()
        
    def generate_hyper(self):
        self.num_tokens = 10392
        self.hidden_size = 4096
        self.vocab_size = 152064
        self.dtype = torch.bfloat16

    def generate_forward_inputs(self):
        hidden = (torch.empty((self.num_tokens, self.hidden_size), dtype=self.dtype,
                              device="cuda").uniform_(-0.5, 0.5).requires_grad_())
        weight = (torch.empty(self.vocab_size, self.hidden_size, dtype=self.dtype,
                              device="cuda").uniform_(-0.5, 0.5).requires_grad_())
        labels = torch.randint(0, self.vocab_size, (self.num_tokens,), device="cuda")
        return hidden, weight, labels
    
    def verify_correctness(self, iterations=5):
        self.cleanup()
        self.generate_hyper()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        hidden, weight, labels = self.generate_forward_inputs()
        print(labels.dtype)

        # torch implementation
        start.record()
        logits = torch.matmul(hidden, weight.T).to(torch.float32)
        end.record()
        torch.cuda.synchronize()
        print(f"torch forward time: {start.elapsed_time(end)} ms")

        vocab_per_split = 1024
        # vocab_per_split = self.vocab_size
        num_splits = (self.vocab_size + vocab_per_split - 1) // vocab_per_split

        # Pad logits with zeros to make it divisible by vocab_per_split
        padded_size = num_splits * vocab_per_split
        if padded_size > self.vocab_size:
            padding = torch.zeros((self.num_tokens, padded_size - self.vocab_size), 
                                 dtype=logits.dtype, device=logits.device)
            padded_logits = torch.cat([logits, padding], dim=1)
        else:
            padded_logits = logits
        # Reshape padded logits to [num_tokens, num_splits, vocab_per_split]
        reshaped_logits = padded_logits.view(self.num_tokens, num_splits, vocab_per_split)
        
        # Max over the vocab_per_split dimension for each split
        split_max = torch.max(reshaped_logits, dim=2)[0]  # [num_tokens, num_splits]
        
        # Extract values from logits based on labels
        torch_logprobs = logits[torch.arange(self.num_tokens, device="cuda"), labels]
        
        # Reshape split_max for broadcasting - keep split dimension separate
        max_expanded = split_max.unsqueeze(2)  # [num_tokens, num_splits, 1]
        
        # Calculate exp(logits - max) with the reshaped tensor, using per-split max values
        exp_logits_reshaped = torch.exp(reshaped_logits - max_expanded)
        
        # Sum over vocab_per_split dimension for each split
        torch_accu = exp_logits_reshaped.sum(dim=2)  # [num_tokens, num_splits]
        
        # Calculate entropy_b using the reshaped tensors
        torch_entropy_b = (reshaped_logits * exp_logits_reshaped).sum(dim=2)  # [num_tokens, num_splits]
        
        # Reshape exp_logits back to original shape for compatibility with rest of code
        # Need to handle the padding when reshaping back to original size
        exp_logits = exp_logits_reshaped.view(self.num_tokens, padded_size)
        if padded_size > self.vocab_size:
            exp_logits = exp_logits[:, :self.vocab_size]  # Remove the padding

        rank = 0
        
        # cute implementation
        _max = torch.empty((self.num_tokens, num_splits), dtype=torch.float32, device="cuda")
        _acc = torch.empty((self.num_tokens, num_splits), dtype=torch.float32, device="cuda")
        _entropy_b = torch.empty((self.num_tokens, num_splits), dtype=torch.float32, device="cuda")
        final_logprobs = torch.empty(self.num_tokens, dtype=torch.float32, device="cuda")
        final_logprobs_scalar = torch.empty((), dtype=torch.float32, device="cuda")

        # gmem_output = torch.empty((self.num_tokens, self.vocab_size), dtype=torch.float, device="cuda")
        gmem_output = None

        start.record()
        with torch.cuda.nvtx.range("forward_mainloop"):
            lce_ext.forward_mainloop(hidden, hidden.stride(0), hidden.stride(1),
                                    weight, weight.stride(0), weight.stride(1),
                                    labels, labels.stride(0),
                                    rank, 
                                    self.num_tokens, self.vocab_size, self.hidden_size,
                                    vocab_per_split,
                                    _max, _max.stride(0), _max.stride(1),
                                    _acc, _acc.stride(0), _acc.stride(1),
                                    _entropy_b, _entropy_b.stride(0), _entropy_b.stride(1),
                                    final_logprobs, final_logprobs_scalar,
                                    gmem_output)
        end.record()
        torch.cuda.synchronize()
        print(f"CUTE forward time: {start.elapsed_time(end)} ms")
        
        if gmem_output is not None:
            print("gmem_output:")
            print(gmem_output)

            print("torch logits")
            print(logits)

            # Find the maximum absolute difference
            logits_float = logits.to(torch.float)
            abs_diff = (gmem_output - logits_float).abs()
            max_diff_value = abs_diff.max().item()
            max_diff_indices = (abs_diff == max_diff_value).nonzero()[0]
            i, j = max_diff_indices
            
            print(f"Maximum absolute difference: {max_diff_value}")
            print(f"At position: [{i}, {j}]")
            print(f"CUTE value: {gmem_output[i, j].item()}")
            print(f"PyTorch value: {logits_float[i, j].item()}")
            
            torch.testing.assert_close(gmem_output, logits_float,
                                    atol=1e-2, rtol=1e-2)

        torch.testing.assert_close(split_max, _max,
                                   atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(torch_logprobs, final_logprobs,
                                   atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(torch_accu, _acc,
                                   atol=1e-1, rtol=1e-1)
        torch.testing.assert_close(torch_entropy_b, _entropy_b,
                                   atol=1e-1, rtol=1e-1)
        print("forward path correctness verified\n")


    def verify_backward_correctness(self, iterations=5):
        self.cleanup()
        self.generate_hyper()

        hidden, weight, labels = self.generate_forward_inputs()
        logits = torch.matmul(hidden, weight.T).to(torch.float32)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        rank = 0

        maximum = logits.max(dim=1)[0]
        exp_logits = torch.exp(logits - maximum.unsqueeze(1))
        accumulate = exp_logits.sum(dim=1)
        pd = torch.nn.functional.softmax(logits, dim=1)
        entropy_b = torch.sum(pd * logits, dim=-1)
        logprobs = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
        logprobs = torch.neg(logprobs)
        grad_entropy = torch.randn((self.num_tokens), dtype=torch.float32, device="cuda")
        grad_logprobs = torch.randn((self.num_tokens), dtype=torch.float32, device="cuda")
        grad_logits = torch.empty((self.num_tokens, self.vocab_size), dtype=self.dtype, device="cuda")

        # gmem_output = torch.empty((self.num_tokens, self.vocab_size), dtype=torch.float32, device="cuda")
        gmem_output = None

        start.record()
        with torch.cuda.nvtx.range("backward_d_logits"):
            lce_ext.backward_d_logits(self.num_tokens, self.hidden_size, self.vocab_size, rank,
                                      hidden, hidden.stride(0), hidden.stride(1),
                                      weight, weight.stride(0), weight.stride(1),
                                      labels, labels.stride(0),
                                      maximum, maximum.stride(0),
                                      accumulate, accumulate.stride(0),
                                      entropy_b, entropy_b.stride(0),
                                      grad_entropy, grad_entropy.stride(0),
                                      grad_logprobs, grad_logprobs.stride(0),
                                      grad_logits, grad_logits.stride(0), grad_logits.stride(1),
                                      gmem_output)
        end.record()
        torch.cuda.synchronize()
        print(f"CUTE backward time: {start.elapsed_time(end)} ms")

        if gmem_output is not None:
            print("gmem_output:")
            print(gmem_output)

            print("torch logits")
            print(logits)
            
            torch.testing.assert_close(gmem_output, logits,
                                       atol=1e-2, rtol=1e-2)
            
        exp_logits = torch.exp(logits - maximum.unsqueeze(1))
        
        # Create mask where labels match the vocabulary indices
        vocab_indices = torch.arange(self.vocab_size, device="cuda")
        vocab_indices = vocab_indices + rank * self.vocab_size
        mask = (labels.unsqueeze(1) == vocab_indices.unsqueeze(0)).to(torch.float32)
        
        # Calculate reciprocal of accumulate for division
        accu_rcp = 1.0 / accumulate
        
        # Calculate d_logits according to the formula
        torch_d_logits = grad_logprobs.unsqueeze(1) * (exp_logits * accu_rcp.unsqueeze(1) - mask)
        torch_d_logits += grad_entropy.unsqueeze(1) * (-1.0 * exp_logits * accu_rcp.unsqueeze(1)) * (logits - entropy_b.unsqueeze(1))
        
        # Convert to the same dtype as grad_logits for comparison
        torch_d_logits = torch_d_logits.to(grad_logits.dtype)

        # Find the maximum absolute difference between torch_d_logits and grad_logits
        abs_diff = torch.abs(torch_d_logits - grad_logits)
        max_abs_diff = torch.max(abs_diff).item()
        max_diff_indices = (abs_diff == max_abs_diff).nonzero()[0]
        i, j = max_diff_indices
        
        print(f"Maximum absolute difference: {max_abs_diff}")
        print(f"At position: [{i}, {j}]")
        print(f"CUTE value: {torch_d_logits[i, j].item()}")
        print(f"PyTorch value: {grad_logits[i, j].item()}")

        print("torch_d_logits")
        print(torch_d_logits)
        print("grad_logits")
        print(grad_logits)
        
        # Verify correctness of gradient computation
        torch.testing.assert_close(torch_d_logits, grad_logits, 
                                  atol=1e-2, rtol=1e-2)
        print("backward path correctness verified")
        
            

if __name__ == "__main__":
    torch.manual_seed(233376)

    test = TestLinearCrossEntropyCUTE()
    test.verify_correctness()
    test.verify_backward_correctness()
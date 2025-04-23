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

        # torch implementation
        start.record()
        logits = torch.matmul(hidden, weight.T)
        end.record()
        torch.cuda.synchronize()
        print(f"torch forward time: {start.elapsed_time(end)} ms")
        
        # cute implementation
        _max = torch.empty((self.num_tokens, 1), dtype=self.dtype, device="cuda")
        _acc = torch.empty((self.num_tokens, 1), dtype=self.dtype, device="cuda")
        _entropy_b = torch.empty((self.num_tokens, 1), dtype=self.dtype, device="cuda")
        final_logprobs = torch.empty(self.num_tokens, dtype=self.dtype, device="cuda")
        final_logprobs_scalar = torch.empty(1, dtype=self.dtype, device="cuda")

        gmem_output = torch.empty((self.num_tokens, self.vocab_size), dtype=torch.float, device="cuda")

        start.record()
        rank = 0
        vocab_per_split = 1024
        with torch.cuda.nvtx.range("forward_mainloop"):
            lce_ext.forward_mainloop(hidden, hidden.stride(0), hidden.stride(1),
                                    weight, weight.stride(1), weight.stride(0),
                                    labels, labels.stride(0),
                                    rank, 
                                    self.num_tokens, self.vocab_size, self.hidden_size,
                                    vocab_per_split,
                                    _max, _max.stride(0), _max.stride(1),
                                    _acc, _acc.stride(0), _acc.stride(1),
                                    _entropy_b, _entropy_b.stride(0), _entropy_b.stride(1),
                                    final_logprobs, final_logprobs_scalar, #None)#,
                                    gmem_output)
        end.record()
        torch.cuda.synchronize()
        print(f"CUTE forward time: {start.elapsed_time(end)} ms")
        
        print("gmem_output:")
        print(gmem_output)

        print("torch logits")
        print(logits)

        torch.testing.assert_close(gmem_output, logits.to(torch.float),
                                   atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    test = TestLinearCrossEntropyCUTE()
    test.verify_correctness()
        
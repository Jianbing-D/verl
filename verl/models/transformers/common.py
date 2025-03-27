import torch
from transformers.models.llama.modeling_llama import CausalLMOutputWithPast
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLCausalLMOutputWithPast

class FusedCausalLMOutputWithPast(CausalLMOutputWithPast):
    log_probs: torch.Tensor
    entropy: torch.Tensor

class FusedQwen2VLCausalLMOutputWithPast(Qwen2VLCausalLMOutputWithPast):
    log_probs: torch.Tensor
    entropy: torch.Tensor
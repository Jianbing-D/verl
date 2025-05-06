import torch
from transformers.models.llama.modeling_llama import CausalLMOutputWithPast
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLCausalLMOutputWithPast

from dataclasses import dataclass
from typing import Optional

@dataclass
class FusedCausalLMOutputWithPast(CausalLMOutputWithPast):
    log_probs: Optional[torch.Tensor] = None
    entropy: Optional[torch.Tensor] = None

@dataclass
class FusedQwen2VLCausalLMOutputWithPast(Qwen2VLCausalLMOutputWithPast):
    log_probs: Optional[torch.Tensor] = None
    entropy: Optional[torch.Tensor] = None
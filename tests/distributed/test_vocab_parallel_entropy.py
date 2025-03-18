import os

os.environ['NCCL_DEBUG'] = 'WARN'

import torch
import torch.distributed

from verl.utils.megatron.tensor_parallel import vocab_parallel_entropy
from verl.utils.torch_functional import logprobs_from_logits, entropy_from_logits

from verl.utils.debug import log_gpu_memory_usage

from megatron.core import mpu

def test_vocab_parallel_entropy():
    # check vocab_parallel_entropy
    mpu.initialize_model_parallel(8, 1)

    batch_size = 2
    seqlen = 128
    vocab_size = 155136

    logits = torch.randn(batch_size * seqlen, vocab_size, device='cuda', requires_grad=True)
    target = torch.randint(low=0, high=vocab_size, size=(batch_size * seqlen,), device='cuda', dtype=torch.int64)

    # broadcast across tp
    torch.distributed.broadcast(logits,
                                mpu.get_tensor_model_parallel_src_rank(),
                                group=mpu.get_tensor_model_parallel_group())
    torch.distributed.broadcast(target,
                                mpu.get_tensor_model_parallel_src_rank(),
                                group=mpu.get_tensor_model_parallel_group())

    tp_rank = mpu.get_tensor_model_parallel_rank()
    vocab_size_per_tp = vocab_size // mpu.get_tensor_model_parallel_world_size()

    # get the local logits of each tp
    vocab_parallel_logits = logits.clone().detach()[:, tp_rank * vocab_size_per_tp:(tp_rank + 1) *
                                                    vocab_size_per_tp].requires_grad_()
    logits.grad = None
    vocab_parallel_logits.grad = None

    log_gpu_memory_usage('begin')
    output_entropy = vocab_parallel_entropy(vocab_parallel_logits)
    log_gpu_memory_usage('after forward')
    grad_output = torch.randn_like(output_entropy)
    output_entropy.backward(grad_output)
    log_gpu_memory_usage('after backward')

    target_entropy = entropy_from_logits(logits)
    torch.testing.assert_close(output_entropy, target_entropy)
    target_entropy.backward(grad_output)
    torch.testing.assert_close(logits.grad[:, tp_rank * vocab_size_per_tp:(tp_rank + 1) * vocab_size_per_tp],
                               vocab_parallel_logits.grad)
    # make sure logits is not altered
    torch.testing.assert_close(logits[:, tp_rank * vocab_size_per_tp:(tp_rank + 1) * vocab_size_per_tp],
                               vocab_parallel_logits)

    if mpu.get_tensor_model_parallel_rank() == 0:
        print('test_vocab_parallel_entropy passes')

    mpu.destroy_model_parallel()
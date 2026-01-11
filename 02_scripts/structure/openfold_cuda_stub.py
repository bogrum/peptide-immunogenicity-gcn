"""
Stub module to replace OpenFold CUDA extensions with CPU fallback.
This allows SwiftMHC to run without building CUDA extensions.
"""

import sys
import torch

class AttnCoreInplaceCudaStub:
    """Stub for attn_core_inplace_cuda module"""

    @staticmethod
    def forward_(attention_logits, batch_size, seq_len):
        """CPU fallback for attention forward pass"""
        # Apply softmax in-place
        attention_logits.copy_(torch.nn.functional.softmax(attention_logits, dim=-1))

    @staticmethod
    def backward_(attention_logits, grad_output, v, batch_size, seq_len, dim):
        """CPU fallback for attention backward pass"""
        # Compute gradients for softmax
        sum_grads = (attention_logits * grad_output).sum(dim=-1, keepdim=True)
        attention_logits.copy_(attention_logits * (grad_output - sum_grads))

# Register the stub module
sys.modules['attn_core_inplace_cuda'] = AttnCoreInplaceCudaStub()

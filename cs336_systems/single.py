import torch
from torch import nn
from cs336_basics.transformer import rope, transformer_block
from torch.utils.checkpoint import checkpoint
'''
x = torch.randn((4, 512, 2560), requires_grad=True,device = 'cuda')
class RMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        device='cuda',
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device))
        self.eps = eps
    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = x * rms
        return self.weight * x
def pack_hook(t):
    shape, dtype, grad_fn = t.shape, t.dtype, t.grad_fn
    print(f"Saving residual: {shape=}, {dtype=}, {grad_fn=}")
    return t
def unpack_hook(t):
    shape, dtype, grad_fn = t.shape, t.dtype, t.grad_fn
    print(f"Loading residual: {shape=}, {dtype=}, {grad_fn=}")
    return t

#原始
# ln = RMSNorm(x.shape[-1])    #x.shape[-1] 表示：获取张量 x 最后一个维度的大小。
ln = torch.compile(RMSNorm(x.shape[-1]))
with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
    y = ln(x)
    y.sum().backward()
    '''
# num_layers for this model is 32
d_model, d_ff, num_heads, theta,seq_len = 2560, 10240, 16, 10000.0,2048
block = transformer_block(d_model=d_model, d_ff=d_ff, num_heads=num_heads,
                          theta = theta,max_seq_len= seq_len,device = 'cuda')
# Fuse as much torch.compile will allow
block = torch.compile(block, fullgraph=True)
x = torch.randn((4, seq_len, d_model), requires_grad=True,device = 'cuda')
# Now logs the number of bytes saved
total_size_bytes = 0
def pack_hook(t):
    if isinstance(t, torch.nn.Parameter): # Skip logging parameters to avoid double counting
        return t
    global total_size_bytes
    shape, dtype, grad_fn = t.shape, t.dtype, t.grad_fn
    total_size_bytes += t.numel() * t.element_size()
    print(f"Saving residual: {shape=}, {dtype=}, {grad_fn=}")
    return t
def unpack_hook(t):
    shape, dtype, grad_fn = t.shape, t.dtype, t.grad_fn
    print(f"Loading residual: {shape=}, {dtype=}, {grad_fn=}")
    return t
# Run forward pass, saving for backward
def four_blocks(x):
    x = block(x)
    x = block(x)
    x = block(x)
    x = block(x)
    return x
def two_blocks(x):
    x = block(x)
    x = block(x)
    return x
def four_blocks_checkpointing(x):
    x = checkpoint(two_blocks, x, use_reentrant=False)
    x = checkpoint(two_blocks, x, use_reentrant=False)
    return x
with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
    y = four_blocks_checkpointing(x)
    print(f"Total size of saved tensors in single TransformerBlock: {total_size_bytes /
        (1024**2):.2f} MiB")
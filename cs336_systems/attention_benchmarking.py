import torch
import torch.nn as nn
from math import sqrt
from torch import Tensor
from einops import einsum, rearrange
from jaxtyping import Float, Bool, Int
from cs336_basics.transformer import scaled_dot_product_attention
from time import perf_counter
from statistics import mean
from itertools import product

w = 5   #warm-up
n = 100
batch_size = 8
d_models = [16,32,64,128]
seq_lens = [256,1024,4096,8192,16384]

torch.cuda.synchronize()
for d_model, seq_len in product(d_models,seq_lens):
    print(f"d_model = {d_model}, seq_len = {seq_len} :")
    baseline_bytes = 0
    before_backward_bytes = 0
    forward_times = []
    backward_times = []
    baseline_bytes = []
    before_backward_bytes = []
    stage = "创建输入"
    try:
        Q = torch.randn((batch_size,seq_len,d_model),device = 'cuda',requires_grad= True)
        K = torch.randn((batch_size,seq_len,d_model),device = 'cuda',requires_grad= True)
        V = torch.randn((batch_size,seq_len,d_model),device = 'cuda',requires_grad= True)
        stage = "warm-up"
        for _ in range(w):
            A = scaled_dot_product_attention(Q,K,V)
            loss = A.sum()
            loss.backward()
            del A, loss
        torch.cuda.synchronize()
        stage = "forward"
        for _ in range(n):
            start = perf_counter()
            A = scaled_dot_product_attention(Q,K,V)
            torch.cuda.synchronize()
            forward_times.append((perf_counter()-start) * 1000)
            del A
        print(f"forward 平均毫秒: {mean(forward_times):.3f} ms")
        stage = "backward"
        for _ in range(n):
            Q.grad = None
            K.grad = None
            V.grad = None
            baseline_bytes.append(torch.cuda.memory_allocated())
            A = scaled_dot_product_attention(Q,K,V)
            loss = A.sum()
            torch.cuda.synchronize()
            start = perf_counter()
            before_backward_bytes.append(torch.cuda.memory_allocated())
            loss.backward()
            torch.cuda.synchronize()
            backward_times.append((perf_counter()-start) * 1000)

            del A, loss
        print(f"backward 平均毫秒: {mean(backward_times):.3f} ms")
        print(f"Backward 前显存: {mean(before_backward_bytes)/ 1024**3:.3f} GiB")
        print(f"Forward 后增加: {(mean(before_backward_bytes) -mean(baseline_bytes) )/ 1024**3:.3f} GiB") 
    except torch.cuda.OutOfMemoryError:
        print(f"OOM on stage={stage}")  
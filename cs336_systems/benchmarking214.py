from cs336_basics.transformer import transformer_lm
from cs336_basics.tool import get_batch
from cs336_basics.optimization import cross_entropy, AdamW
from dataclasses import asdict, dataclass
from typing import Literal
from pathlib import Path
import torch
from time import perf_counter
import statistics
import gc


size = "medium"  
MODEL_CONFIGS = {
    "small": {
        "d_model": 768,
        "d_ff": 3072,
        "num_layers": 12,
        "num_heads": 12,
    },
    "medium": {
        "d_model": 1024,
        "d_ff": 4096,
        "num_layers": 24,
        "num_heads": 16,
    },
    "large": {
        "d_model": 1280,
        "d_ff": 5120,
        "num_layers": 36,
        "num_heads": 20,
    },
    "xl": {
        "d_model": 2560,
        "d_ff": 10240,
        "num_layers": 32,
        "num_heads": 32,
    },
    "10B":{
        "d_model": 4608,
        "d_ff": 12288,
        "num_layers": 50,
        "num_heads": 36,
    }
}
@dataclass
class TrainConfig:
    # 路径
    train_path: str = "data/train_data.bin"
    validation_path: str = "data/validation_data.bin"
    out_dir: str = "result/"

    # 数据类型
    data_dtype: str = "uint16"
    model_dtype: torch.dtype = torch.float32

    # 模型
    vocab_size: int = 10_000
    seq_len: int = 512
    d_model: int = MODEL_CONFIGS[size]["d_model"]
    d_ff: int = MODEL_CONFIGS[size]["d_ff"]
    num_layers: int = MODEL_CONFIGS[size]["num_layers"]
    num_heads: int = MODEL_CONFIGS[size]["num_heads"]
    rope_theta: float = 10_000.0

    # 优化器与学习率调度
    lr: float = 3e-4
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    alpha_max: float = 3e-4
    alpha_min: float = alpha_max / 10
    total_steps: int = 5000
    T_w: int = 0
    T_c: int = total_steps
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    # 训练过程
    batch_size: int = 4
    device: Literal["auto", "cpu", "cuda", "mps"] = "cuda"
    seed: int = 0
    log_interval: int = 20    #每隔多少步记录一次训练日志
    eval_interval: int = 200   #每隔几步进行一次验证
    eval_batches: int = 10    #每次验证使用多少个 batch
    checkpoint_interval: int = 0    #每隔多少步覆盖保存一次最新 checkpoint
    milestone_interval: int = 0     #每隔多少步额外保留一个不会被覆盖的 checkpoint
    # 是否从一个已有 checkpoint 恢复训练，以及要从哪个 checkpoint 文件恢复。
    resume_from: str | None = None

def run_step(model, inputs, targets, optimizer,mode):
    if mode not in {"forward", "forward_backward",'full_step'}:
        raise ValueError(f"Unknown mode: {mode}")
    if mode in {'forward_backward','full_step'}:
        model.zero_grad(set_to_none = True)
    
    with torch.cuda.nvtx.range("forward"):
        logits = model(inputs)

    if mode == "forward":
        return

    with torch.cuda.nvtx.range("loss"):
        loss = cross_entropy(logits, targets)

    with torch.cuda.nvtx.range("backward"):
        loss.backward()

    if mode == "full_step":
        with torch.cuda.nvtx.range("optimizer"):
            optimizer.step()
    
    return
def main(config: TrainConfig) -> None:
    w = 5    #warmup数量
    n = 10
    token_id = torch.randint(low = 0, high = config.vocab_size,size = (config.batch_size,config.seq_len + 1),
                             dtype = torch.long,device = "cuda")
    inputs = token_id[:,:-1]
    targets = token_id[:,1:]
    modes = ['forward','forward_backward','full_step']
    print(f"Size: {size}")

    for mode in modes:
        model = None
        optimizer = None
        model = transformer_lm(
            vocab_size = config.vocab_size,
            seq_len = config.seq_len,
            num_layers = config.num_layers,
            d_model = config.d_model,
            num_heads = config.num_heads,
            d_ff = config.d_ff,
            theta = config.rope_theta,
            device = config.device,
            dtype = config.model_dtype
        )
        optimizer = AdamW(
            model.parameters(),
            lr = 3e-4,
            betas = config.betas,
            eps = config.eps,
            weight_decay = config.weight_decay
        )
        for _ in range(w):
            run_step(model,inputs,targets,optimizer,mode = mode)
            torch.cuda.synchronize("cuda")
        with torch.cuda.nvtx.range("benchmark"):
            for _ in range(n):
                run_step(model,inputs,targets,optimizer,mode = mode)
                torch.cuda.synchronize("cuda")
        if optimizer is not None:
            del optimizer
        if model is not None:
            del model
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main(TrainConfig())
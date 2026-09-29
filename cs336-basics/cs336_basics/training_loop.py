import numpy as np
from cs336_basics.bpe import train_bpe,Tokenizer
from cs336_basics.transformer import transformer_lm
from cs336_basics.optimization import AdamW ,cross_entropy,gradient_clipping, learning_rate_schedule
from cs336_basics.tool import get_batch,save_checkpoint, load_checkpoint
from cs336_basics.tool import log_jsonl,evaluate,make_fixed_batches
import torch
import torch.nn as nn
from torch import Tensor
import os
import json, time
from dataclasses import asdict, dataclass
from typing import Literal
from pathlib import Path


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
    seq_len: int = 256
    d_model: int = 512
    d_ff: int = 1344
    num_layers: int = 4
    num_heads: int = 16
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
    batch_size: int = 32
    device: Literal["auto", "cpu", "cuda", "mps"] = "cuda"
    seed: int = 0
    log_interval: int = 20    #每隔多少步记录一次训练日志
    eval_interval: int = 200   #每隔几步进行一次验证
    eval_batches: int = 10    #每次验证使用多少个 batch
    checkpoint_interval: int = 0    #每隔多少步覆盖保存一次最新 checkpoint
    milestone_interval: int = 0     #每隔多少步额外保留一个不会被覆盖的 checkpoint
    # 是否从一个已有 checkpoint 恢复训练，以及要从哪个 checkpoint 文件恢复。
    resume_from: str | None = None


def main(config: TrainConfig) -> None:
    # 1️⃣设置文件路径
    out_dir = Path(config.out_dir)  #把字符串转换成path对象
    out_dir.mkdir(parents=True,exist_ok=True)
    config_path = out_dir / "config.json"    #配置文件  
    metrics_path = out_dir / "metrics.jsonl"   #训练数据文件
    last_checkpoint_path = out_dir / "ckpt_last.pt"   #反复覆盖的最新 checkpoint 路径
    final_checkpoint_path = out_dir / "ckpt_final.pt"  #训练全部完成后的最终 checkpoint 路径
    #保存到config.json
    with open(config_path,"w",encoding="utf-8",) as f:
        json.dump(asdict(config),f,indent=2,ensure_ascii=False,default=str)
    
    #2️⃣.设置随机种子
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    #3️⃣.读取数据
    #以 内存映射的方式读取训练集和验证集的 token ID
    train_data = np.memmap(config.train_path,dtype = np.dtype(config.data_dtype),mode = "r")   #将train data的bin转化为int list
    validation_data = np.memmap(config.validation_path,dtype = np.dtype(config.data_dtype),mode ="r")
    #4️⃣.创建模型
    model = transformer_lm(
        config.vocab_size,
        config.seq_len,
        config.num_layers,
        config.d_model,
        config.num_heads,
        config.d_ff,
        theta = config.rope_theta,
        device = config.device,
        dtype = config.model_dtype
    )
    model.train()     #进入训练模式
    n_params = sum(p.numel() for p in model.parameters()) #统计总参数数量
    print(f"[model] {n_params:,} parameters")
    #5️⃣.创建优化器
    optimizer = AdamW(
        model.parameters(),
        config.lr,
        config.betas,
        config.eps,
        config.weight_decay
    )
    #6️⃣是否需要恢复checkpoint
    #默认无需恢复checkpoint，从0开始训练，但如果需要恢复，则从恢复点开始训练
    start_step = 0
    if config.resume_from:
        start_step = load_checkpoint(config.resume_from, model, optimizer)
        print(f"[resume] 从 {config.resume_from} 恢复,从第 {start_step} 步继续")

    #7️⃣从验证集中预先采样的一组固定 batch
    validation_batches = make_fixed_batches(validation_data, config.batch_size,
                        config.seq_len,config.device, config.eval_batches, config.seed + 1)
    #8️⃣.正式开始训练并计时
    t0 = time.perf_counter()
    tokens_per_step = config.batch_size * config.seq_len
    loss_value = float("nan")
    for step in range(start_step, config.total_steps):
        # 1. 计算当前学习率，写进 param_groups
        lr = learning_rate_schedule(step, config.alpha_max, config.alpha_min, config.T_w, config.T_c)
        for group in optimizer.param_groups:
            #大部分情形只有一个参数组
            group["lr"] = lr
        # 2.清除旧梯度
        optimizer.zero_grad(set_to_none=True)
        # 3. 采样一个batch
        inputs, targets = get_batch(train_data, config.batch_size,config.seq_len, config.device)
        # 4. 前向传播 + 计算loss + 反向传播
        logits = model(inputs)
        loss = cross_entropy(logits, targets)
        loss.backward()
        # 5. 梯度裁剪
        norm = gradient_clipping(model.parameters(),config.grad_clip)
        # 6. AdamW更新参数
        optimizer.step()

        
        # 7. 周期性完成日志、验证、checkpoint等记录任务
        #need_log:当前训练 step 是否需要打印并保存训练日志
        need_log = None
        if config.log_interval > 0:
            need_log = ((step+1) % config.log_interval == 0) or ((step+1) == config.total_steps)
        if need_log:
            loss_value = loss.item()  #储存loss数值
            elapsed = time.perf_counter() - t0  #到目前为止训练的总时长
            done = step - start_step + 1  #已进行的训练次数
            print(f"step {step:6d} | loss {loss_value:8.4f} | lr {lr:.3e} | gnorm {norm:8.3f}"
                  f" | {elapsed:7.1f}s | {done * tokens_per_step / max(elapsed, 1e-9):9,.0f} tokens/s")
            log_jsonl(metrics_path, {"step": step, "train_loss": loss_value, "lr": lr,
                                     "grad_norm": norm, "elapsed": elapsed})

        #定期验证模型
        if config.eval_interval > 0 and ((step + 1) % config.eval_interval == 0 or step + 1 == config.total_steps):
            val_loss = evaluate(model, validation_batches)
            print(f"step {step:6d} | validation loss {val_loss:8.4f}")
            log_jsonl(metrics_path, {"step": step, "val_loss": val_loss, "elapsed": time.perf_counter() - t0})
        #记录checkpoint
        if config.checkpoint_interval > 0 and (step + 1) % config.checkpoint_interval == 0:
            # 定期覆盖同一个文件（崩溃恢复用）
            save_checkpoint(model, optimizer, step + 1, last_checkpoint_path)
            if config.milestone_interval > 0 and (step + 1) % config.milestone_interval == 0:
                # 关键节点另存（事后分析用）
                save_checkpoint(model, optimizer, step + 1, os.path.join(config.out_dir, f"ckpt_{step + 1}.pt"))
    save_checkpoint(model, optimizer, config.total_steps, final_checkpoint_path)
    final_val = evaluate(model, validation_batches)
    total_time = time.perf_counter() - t0
    print(f"[done] {config.total_steps - start_step} steps in {total_time:.1f}s"
          f" | final train loss {loss_value:.4f} | final validation loss {final_val:.4f}")
    log_jsonl(metrics_path, {"step": config.total_steps, "final_validation_loss": final_val, "elapsed": total_time})






    return
if __name__ == '__main__':
    main(TrainConfig())
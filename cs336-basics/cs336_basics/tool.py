import torch
import numpy as np
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Bool, Int
import os
import json
from typing import BinaryIO,IO
from pathlib import Path
from cs336_basics.bpe import Tokenizer
from optimization import cross_entropy
from itertools import islice

def get_batch(
        token_ids: Int[np.ndarray,"num_tokens"],
        batch_size: int,
        seq_len: int,
        device = None
) -> tuple[Int[Tensor,"batch_size seq_len"], Int[Tensor,"batch_size seq_len"]]:
    """get_batch 输入使用 NumPy array，主要是因为这里的 dataset 代表完整的 token 数据集，
    它可能非常大；PyTorch Tensor 则更适合作为已经采样出来、即将送进模型的小 batch。"""
    """
    随机在整个token_ids上采样batch_size次，每次的长度seq_len
    """
    legal_start = len(token_ids) - seq_len
    start_indices = np.random.randint(low = 0, high = legal_start,size = batch_size)
    off_set = np.arange(stop = seq_len)
    inputs_np = token_ids[start_indices[:,None] + off_set[None,:]]
    targets_np = token_ids[start_indices[:,None] + off_set[None,:] + 1]
    inputs = torch.tensor(inputs_np, dtype = torch.long,device = device)   #这里是小写工厂函数
    targets = torch.tensor(targets_np, dtype = torch.long,device = device)
    return inputs, targets

def save_checkpoint(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        iteration: int,
        out: str | os.PathLike | BinaryIO | IO[bytes]
):
    state_model = model.state_dict()
    state_opt = optimizer.state_dict()
    obj = {'model': state_model,'opt': state_opt,'iteration': iteration}
    torch.save(obj,out)

def load_checkpoint(
        src: str | os.PathLike | BinaryIO | IO[bytes],
        model: nn.Module,
        optimizer: torch.optim.Optimizer
):
    obj = torch.load(src)
    model.load_state_dict(obj['model'])
    optimizer.load_state_dict(obj['opt'])
    return obj['iteration']

def save_bpe(
        vocab: dict[int,bytes],
        merges: list[tuple[bytes,bytes]],
        special_tokens: list[str],
        output: str | os.PathLike | BinaryIO | IO[bytes]
) -> None:
    #把vocab,merges,special_tokens保存为json文件，
    #由于json文件无法保存bytes，所以要把bytes转化为hex保存进文件中
    #JSON object 的键只能是字符串，不能真正保存整数键
    output = Path(output)
    data = {
        "vocab":{str(token_id): token_bytes.hex() for token_id, token_bytes in vocab.items()},
        "merges":[(left.hex(),right.hex())for left,right in merges],
        "special_tokens": special_tokens
    }
    with output.open(mode="w",encoding = "utf-8") as f:
        json.dump(data,f,ensure_ascii=False,indent = 2)
        f.write("\n")
    return

def iter_documents(
    file,  #已经以文本模式打开的文件对象
    delimiter="<|endoftext|>",
    chunk_size=1024 * 1024,
):
    '''
    从一个很大的文本文件中分块读取内容，并按照 <|endoftext|> 将文件逐篇切分成文档。
    定义一个生成器函数。调用它时不会立即读取文件，只有开始遍历返回的生成器时，函数才真正执行。
    '''
    buffer = ""

    while chunk := file.read(chunk_size):    #海象运算符 :=，它同时完成赋值和条件判断
        buffer += chunk
        parts = buffer.split(delimiter)

        for document in parts[:-1]:
            yield document + delimiter

        buffer = parts[-1]   #最后一个未完成部分

    if buffer:
        yield buffer

def tokenize_text_to_bin(
    tokenizer,
    input_path,
    output_path,
    dtype,
    batch_size=1_000_000,
):
    """
    从大型文本文件中逐篇读取文档。
    使用 tokenizer 将文本转换成 token ID。
    每次收集固定数量的 token ID。
    将它们以 uint16 二进制格式分批写入磁盘。
    dtype:决定每个 token ID 以什么数字格式保存到二进制文件中,uint16的可表示的范围是 0～65535
    返回总 token 数量。
    """
    total_tokens = 0

    with open(input_path, "r", encoding="utf-8") as source:
        documents = iter_documents(source)  #生成器
        token_iterator = tokenizer.encode_iterable(documents)

        with open(output_path, "wb") as output:
            while True:
                token_batch = np.fromiter(
                    islice(token_iterator, batch_size),
                    dtype=dtype,
                )

                if token_batch.size == 0:
                    break

                token_batch.tofile(output)
                total_tokens += token_batch.size
    return total_tokens

def make_fixed_batches(
        data: Int[np.ndarray,"num_tokens"], #通常是验证集memmap
        batch_size: int,
        context_length: int,
        device: torch.device,
        num_batches: int,
        seed: int,
) -> list[tuple[Tensor, Tensor]]:
    """提前从验证集随机采样固定的一组 batch。
    之后每次验证都使用同样的数据，避免验证曲线因为每次随机样本不同而抖动。
    """
    state = np.random.get_state()
    np.random.seed(seed)
    batches = [get_batch(data, batch_size, context_length, device) for _ in range(num_batches)]
    np.random.set_state(state)
    return batches

def log_jsonl(path: str, record: dict) -> None:
    """每条日志追加一行 JSON。Section 6/7 画曲线时直接读这个文件，不必去 grep stdout。"""
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")

@torch.no_grad()
def evaluate(model: nn.Module, batches: list[tuple[Tensor, Tensor]]) -> float:
    """在固定验证 batch 上计算平均交叉熵，并禁止构建反向传播计算图"""
    model.eval()
    total = 0.0
    for inputs, targets in batches:
        total += cross_entropy(model(inputs), targets).item()
    model.train()
    return total / len(batches)
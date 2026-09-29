
import numpy as np
from cs336_basics.bpe import train_bpe,Tokenizer
from cs336_basics.tool import save_bpe,data_loading,tokenize_text_to_bin
from time import perf_counter

def main():
    #1.1训练bpe，得到vocab,merges和special_tokens
    input_path_1="data/TinyStoriesV2-GPT4-train.txt"
    input_path_2 = "data/TinyStoriesV2-GPT4-valid.txt"

    tokenizer_json_path = "data/tokenizer.json"

    train_path = 'data/train_data.bin'

    validation_path = 'data/validation_data.bin'

    special_tokens=["<|endoftext|>"]

    start = perf_counter()
    vocab, merges = train_bpe(
        input_path=input_path_1,
        vocab_size=10_000,
        special_tokens=special_tokens,
        num_processes=24
)
    elapsed = perf_counter() - start
    print(f"bpe耗时：{elapsed:.6f} 秒")
    #1.2将vocab,merges和special_tokens保存到json文件中，并生成对应的tokenizer

    start = perf_counter()
    save_bpe(vocab, merges, special_tokens,tokenizer_json_path)
    tokenizer = Tokenizer(
        vocab = vocab,
        merges = merges,
        special_tokens= special_tokens
    )
    elapsed = perf_counter() - start
    print(f"保存tokenizer数据耗时：{elapsed:.6f} 秒")
    #2.1将.txt文本转换为token id(用bin方法：.bin 方法的核心是：
    #每次读取一小段文本，立刻编码成 token IDs，然后把这些整数以二进制形式追加到磁盘，不在内存中保存完整 token 数组。)
    #数据类型默认是int16,耗时太大，所以要指定为np.uint16
    start = perf_counter()

    tokenize_text_to_bin(tokenizer,input_path_1,train_path,dtype = np.uint16)

    elapsed = perf_counter() - start
    print(f"保存训练token id耗时：{elapsed:.6f} 秒")

    start = perf_counter()

    tokenize_text_to_bin(tokenizer,input_path_2,validation_path,dtype = np.uint16)

    elapsed = perf_counter() - start
    print(f"保存验证token id耗时：{elapsed:.6f} 秒")
if __name__ == '__main__':
    main()
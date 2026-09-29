import os
from typing import BinaryIO
import regex as re
from collections import Counter
from collections import defaultdict
from multiprocessing import Pool
from functools import lru_cache  #缓存工具
import json

from collections.abc import Iterable,Iterator

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def find_chunk_boundaries(
        
    file: BinaryIO,   #以二进制模式打开的文件对象
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    将文本分成若干个chunk，每个chunk以特殊token开头(第一个chunk除外)
    把一个大文本文件划分成多个适合多进程处理的区间
    返回每一个chunk的第一个byte的位置
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"
    
    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    
    chunk_size = file_size // desired_num_chunks
    
    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size
    
    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    #接下来要改进boundary以使每个chunk的开头是特殊token

    for i in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[i]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[i] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:    #find函数的-1表示没有找到
                chunk_boundaries[i] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))

def count_chunk(
        input_path: str,
        start: int,
        end: int,
        special_tokens: list[str]
) -> Counter:
    '''
    统计函数
    读取训练文件的一个指定区间，去掉特殊 token，对普通文本进行预分词，然后统计每个 pre-token 出现了多少次
    '''
    with open(input_path, 'rb') as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        #接下来几行是把special tokens从文本中切出来，不进行bpe
        if special_tokens == None:
            parts= chunk
        else:
            patterns = "|".join(re.escape(special) for special in special_tokens)
            parts = re.split(patterns, chunk)
        counter = Counter()
        # Run pre-tokenization on your chunk and store the counts for each pre-token
        for part in parts:
            for pre_token in re.finditer(PAT,part):   #将文本分成分词(str)
                str_word = pre_token.group().encode('utf-8')  #存入的是bytes
                word = tuple(bytes([x]) for x in str_word)
                counter[word] += 1  
    return counter

def train_bpe(
        input_path: str,
        vocab_size: int,
        special_tokens: list[str],
        num_processes: int = 24,
) -> tuple[dict[int,bytes],list[tuple[bytes,bytes]]]:
    """
    假设vocabulary的size一定够用
    """
    frequency = Counter()       #每个word出现的频率
    pairs = Counter()     #每个pair出现的频率
    pair_word = defaultdict(set)  #防止aaaa使得出现相同的word
    vocab = {i : bytes([i]) for i in range(256)}    #bytes函数的用法
    for k,v in enumerate(special_tokens):
        vocab[256 + k] = v.encode('utf-8')
    index = 256 + len(special_tokens)   #记录下一个新词的索引

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        #并行构建frequency
        task = [(input_path, start, end, special_tokens) for start, end in zip(boundaries[:-1],boundaries[1:])]
        with Pool(processes = num_processes) as pool:
            counters = pool.starmap(count_chunk, task)
            for counter in counters:
                frequency.update(counter)
    #构建pair和pair-word
    for word in frequency:
        if len(word) == 1:
            continue
        for i in range(len(word)-1):
            pair = (word[i],word[i+1])
            pairs[pair] += frequency[word]
            pair_word[pair].add(word)
    #选出出现次数最多的pair
    merges = []
    while index < vocab_size:
        max_pair = max(pairs,key = lambda pair: (pairs[pair],pair))   #关键，可以进行改进！！！
        vocab[index] = max_pair[0] + max_pair[1]   #更新vocab
        merges.append(max_pair)     #更新merges
        #merge，修改frequency，pair，pair_word
        for word in pair_word[max_pair].copy():  #因为我们要修改pair_word
            n = len(word)
            new_word = []
            count = frequency[word]
            for i in range(len(word)-1):
                old_pair = (word[i],word[i+1])
                pairs[old_pair] -= count
            i = 0
            while i < n-1:
                pair = (word[i],word[i+1])
                if pair != max_pair:
                    new_word.append(word[i])
                    i += 1
                else:
                    new_word.append(word[i]+word[i+1])
                    i += 2
            if i == n-1:
                new_word.append(word[n-1])
            new_word = tuple(new_word)
            frequency[new_word] = count
            frequency.pop(word,None)
            for i in range(len(new_word)-1):
                new_pair = (new_word[i],new_word[i+1])
                pairs[new_pair] += count
            #由于word变成了new_word,所以word中的字节对pair_word也要变成新的
            for old_pair in zip(word[:-1],word[1:]):
                pair_word[old_pair].discard(word)
                if pairs[old_pair] == 0:
                    pairs.pop(old_pair, None)
                    pair_word.pop(old_pair, None)
            for new_pair in zip(new_word[:-1],new_word[1:]):
                pair_word[new_pair].add(new_word)
        pairs.pop(max_pair,None)
        index += 1

    return vocab, merges

class Tokenizer():
    def __init__(self,vocab,merges,special_tokens = None):
        self.vocab = vocab
        self.merges = merges
        self.max = len(self.merges)   #rank最大也不可能达到的值
        self.special_tokens = special_tokens
        self.reverse_vocab = {token: ids for ids,token in self.vocab.items()}
        self.reverse_merge = defaultdict(lambda: self.max)
        for i in range(len(merges)):
            self.reverse_merge[merges[i]] = i    #记录每个merge的rank，rank越小优先级越高

        self.patterns = "|".join(re.escape(special)
                for special in sorted(self.special_tokens, key=len, reverse=True))
    @classmethod
    def from_files(
            cls,
            filepath: str,
            special_tokens: list[str] = None
    ):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab = {int(token_id): bytes.fromhex(token_hex)
            for token_id, token_hex in data["vocab"].items()}
        merges = [(bytes.fromhex(left), bytes.fromhex(right))
            for left, right in data["merges"]]
        special_tokens = data.get("special_tokens", None)
        return cls(
            vocab=vocab,
            merges=merges,
            special_tokens=special_tokens,
        )
    #对pre_token级别进行缓存
    @lru_cache(maxsize=100_000)
    def encode_word(self,pre_token:str) -> list[int]:  #对单个pre-token进行encode，加入cache
        word = [bytes([x]) for x in pre_token.encode('utf-8')]
        while len(word) > 1:
            ranks = {pair:self.reverse_merge[pair] for pair in zip(word[:-1],word[1:])}
            merge = min(ranks,key = ranks.get)
            if ranks[merge] == self.max:   #没有可以merge的了
                break
            else:
                new_word = []
                n = len(word)
                i = 0
                while i < n-1:
                    pair = (word[i],word[i+1])
                    if pair != merge:
                        new_word.append(word[i])
                        i += 1
                    else:
                        new_word.append(word[i]+word[i+1])
                        i += 2
                if i == n-1:
                    new_word.append(word[n-1])
                word = tuple(new_word)
        return [self.reverse_vocab[x] for x in word]
    def encode(self, text: str) -> list[int]:
        tokens = []
        if not self.special_tokens:
            for pre_token in re.finditer(PAT,text):
                pre_token = pre_token.group()
                tokens.extend(self.encode_word(pre_token))
        else:
            parts = re.split(f"({self.patterns})", text) #保留特殊token
            for part in parts:
                if part in self.special_tokens:
                    tokens.append(self.reverse_vocab[part.encode('utf-8')])
                else:
                    for pre_token in re.finditer(PAT,part):
                        pre_token = pre_token.group()
                        tokens.extend(self.encode_word(pre_token))
        return tokens
    def encode_iterable(self,iterable:Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)
    def decode(self, ids: list[int]) -> str:
        b_text = b''    #必须先合并byte，再解码，不能逐个解码
        for i in ids:
            b_text += self.vocab[i]
        text = b_text.decode('utf-8',errors="replace") #遇到无法解码的字节时，用默认U+FFFD替换，继续解码
        return text


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BPE (Byte Pair Encoding) 分词器核心模块 / BPE (Byte Pair Encoding) Tokenizer Core Module
实现字节级BPE分词器的训练和编码解码功能
Implements byte-level BPE tokenizer training and encoding/decoding functionality
"""

import collections
import os
from typing import Dict, List, Tuple


def train_bpe(input_path: str, vocab_size: int, special_tokens: List[str], max_merges: int = None) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    训练字节级BPE分词器 / Train byte-level BPE tokenizer
    
    Args:
        input_path: 训练数据文本文件的路径 / Path to training data text file
        vocab_size: 最终词汇表大小（包括初始字节词汇、合并产生的词汇和特殊令牌）/ Final vocabulary size (including initial byte vocabulary, merge-generated vocabulary, and special tokens)
        special_tokens: 要添加到词汇表中的特殊令牌列表 / List of special tokens to add to vocabulary
        max_merges: 最大合并次数，如果为None则不限制 / Maximum number of merges, if None then no limit
    
    Returns:
        vocab: 词汇表，映射从token ID到bytes / Vocabulary mapping from token ID to bytes
        merges: BPE合并操作列表，按创建顺序排列 / BPE merge operations list, ordered by creation sequence
    """
    
    # 1. 初始化词汇表（256个字节 + 特殊令牌）/ Initialize vocabulary (256 bytes + special tokens)
    vocab = {}
    next_id = 0
    
    # 添加基础字节词汇 (0-255) / Add basic byte vocabulary (0-255)
    for i in range(256):
        vocab[next_id] = bytes([i])
        next_id += 1
    
    # 添加特殊令牌 / Add special tokens
    special_token_bytes = []
    for token in special_tokens:
        token_bytes = token.encode('utf-8')
        special_token_bytes.append(token_bytes)
        vocab[next_id] = token_bytes
        next_id += 1
    
    # 2. 读取训练数据并统计词频 / Read training data and count word frequencies
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # 将文本转换为UTF-8字节序列 / Convert text to UTF-8 byte sequence
    byte_data = text.encode('utf-8')
    
    # 3. 初始化词汇：将文本拆分为单词，每个单词拆分为字节 / Initialize vocabulary: split text into words, each word into bytes
    words = text.split()
    word_freqs = collections.Counter(words)
    
    # 构建初始词汇统计（字节序列）/ Build initial vocabulary statistics (byte sequences)
    vocab_stats = {}
    for word, freq in word_freqs.items():
        byte_word = word.encode('utf-8')
        # 将单词表示为字节列表 / Represent word as byte list
        tokens = [bytes([b]) for b in byte_word]
        vocab_stats[tuple(tokens)] = freq
    
    # 4. BPE训练循环 / BPE training loop
    merges = []
    merge_count = 0
    
    while len(vocab) < vocab_size:
        # 检查是否达到最大合并次数限制 / Check if maximum merge count limit reached
        if max_merges is not None and merge_count >= max_merges:
            break
            
        # 统计所有相邻字节对的出现频率 / Count frequency of all adjacent byte pairs
        pair_freqs = collections.Counter()
        
        for tokens, freq in vocab_stats.items():
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pair_freqs[pair] += freq
        
        if not pair_freqs:
            break  # 没有更多可以合并的对 / No more pairs to merge
        
        # 找到频率最高的字节对（如果有多个相同频率的，选择最后一个）/ Find the byte pair with highest frequency (if multiple have same frequency, choose the last one)
        max_freq = max(pair_freqs.values())
        # 找到所有频率等于最大频率的字节对 / Find all byte pairs with frequency equal to maximum
        best_pairs = [pair for pair, freq in pair_freqs.items() if freq == max_freq]
        # 选择最后一个字节对 / Choose the last byte pair
        best_pair = best_pairs[-1] if best_pairs else None
        
        if best_pair is None:
            break
        
        # 检查是否达到词汇表大小限制 / Check if vocabulary size limit reached
        if len(vocab) >= vocab_size:
            break
        
        # 创建新的合并token / Create new merge token
        new_token = best_pair[0] + best_pair[1]
        
        # 添加到词汇表 / Add to vocabulary
        vocab[next_id] = new_token
        next_id += 1
        
        # 记录合并操作 / Record merge operation
        merges.append(best_pair)
        merge_count += 1
        
        # 更新词汇统计：合并所有出现的best_pair / Update vocabulary statistics: merge all occurrences of best_pair
        new_vocab_stats = {}
        for tokens, freq in vocab_stats.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                if (i < len(tokens) - 1 and 
                    tokens[i] == best_pair[0] and 
                    tokens[i + 1] == best_pair[1]):
                    new_tokens.append(new_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_vocab_stats[tuple(new_tokens)] = freq
        
        vocab_stats = new_vocab_stats
    
    return vocab, merges


class BPETokenizer:
    """BPE分词器类 / BPE Tokenizer Class"""
    
    def __init__(self, vocab: Dict[int, bytes] = None, merges: List[Tuple[bytes, bytes]] = None):
        self.vocab = vocab or {}
        self.merges = merges or []
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self._trained = len(self.vocab) > 0  # 标记是否已训练 / Mark whether trained
    
    def train(self, input_path: str, vocab_size: int, special_tokens: List[str], max_merges: int = None):
        """训练BPE分词器 / Train BPE tokenizer"""
        self.vocab, self.merges = train_bpe(input_path, vocab_size, special_tokens, max_merges)
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self._trained = True  # 标记为已训练 / Mark as trained
    
    def encode(self, text: str) -> List[int]:
        """将文本编码为token IDs / Encode text to token IDs"""
        if not self._trained:
            raise ValueError("Tokenizer需要先训练或加载词汇表 / Tokenizer needs to be trained or loaded with vocabulary first")
        
        # 将文本转换为字节序列 / Convert text to byte sequence
        byte_data = text.encode('utf-8')
        tokens = [bytes([b]) for b in byte_data]
        
        # 应用BPE合并规则 / Apply BPE merge rules
        for merge_pair in self.merges:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if (i < len(tokens) - 1 and 
                    tokens[i] == merge_pair[0] and 
                    tokens[i + 1] == merge_pair[1]):
                    new_tokens.append(merge_pair[0] + merge_pair[1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        
        # 转换为token IDs / Convert to token IDs
        token_ids = []
        for token in tokens:
            if token in self.reverse_vocab:
                token_ids.append(self.reverse_vocab[token])
            else:
                # 未知token，使用第一个特殊token / Unknown token, use first special token
                special_tokens = [k for k, v in self.vocab.items() if v.startswith(b'<')]
                if special_tokens:
                    token_ids.append(special_tokens[0])
                else:
                    token_ids.append(0)  # 默认使用字节0 / Default to byte 0
        
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """将token IDs解码为文本 / Decode token IDs to text"""
        if not self._trained:
            raise ValueError("Tokenizer需要先训练或加载词汇表 / Tokenizer needs to be trained or loaded with vocabulary first")
        
        # 将token IDs转换为字节序列 / Convert token IDs to byte sequence
        byte_data = b''
        for token_id in token_ids:
            if token_id in self.vocab:
                byte_data += self.vocab[token_id]
        
        # 解码为文本 / Decode to text
        try:
            return byte_data.decode('utf-8')
        except UnicodeDecodeError:
            return byte_data.decode('utf-8', errors='replace')
    
    def save(self, vocab_path: str, merges_path: str):
        """保存词汇表和合并规则 / Save vocabulary and merge rules"""
        # 保存词汇表 / Save vocabulary
        with open(vocab_path, 'w', encoding='utf-8') as f:
            for token_id, token_bytes in self.vocab.items():
                token_str = token_bytes.decode('utf-8', errors='replace')
                f.write(f"{token_id}\t{repr(token_str)}\n")
        
        # 保存合并规则 / Save merge rules
        with open(merges_path, 'w', encoding='utf-8') as f:
            for i, (token1, token2) in enumerate(self.merges):
                token1_str = token1.decode('utf-8', errors='replace')
                token2_str = token2.decode('utf-8', errors='replace')
                f.write(f"{i}\t{repr(token1_str)}\t{repr(token2_str)}\n")
    
    def load(self, vocab_path: str, merges_path: str):
        """加载词汇表和合并规则 / Load vocabulary and merge rules"""
        # 加载词汇表 / Load vocabulary
        self.vocab = {}
        with open(vocab_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t', 1)
                if len(parts) == 2:
                    token_id = int(parts[0])
                    token_str = eval(parts[1])  # 使用eval转换repr字符串 / Use eval to convert repr string
                    self.vocab[token_id] = token_str.encode('utf-8')
        
        # 加载合并规则 / Load merge rules
        self.merges = []
        with open(merges_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    token1_str = eval(parts[1])
                    token2_str = eval(parts[2])
                    self.merges.append((token1_str.encode('utf-8'), token2_str.encode('utf-8')))
        
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self._trained = len(self.vocab) > 0


def create_sample_training_data():
    """创建示例训练数据 / Create sample training data"""
    sample_text = """
    The quick brown fox jumps over the lazy dog.
    Hello world! This is a sample text for BPE training.
    Machine learning and natural language processing.
    Tokenization is an important step in text processing.
    Byte Pair Encoding is a subword tokenization algorithm.
    """
    
    # 创建临时文件 / Create temporary file
    with open('sample_training_data.txt', 'w', encoding='utf-8') as f:
        f.write(sample_text)
    
    return 'sample_training_data.txt'


def test_bpe_tokenizer():
    """测试BPE分词器 / Test BPE tokenizer"""
    # 创建示例数据 / Create sample data
    train_file = create_sample_training_data()
    
    # 训练分词器 / Train tokenizer
    tokenizer = BPETokenizer()
    special_tokens = ['<pad>', '<unk>', '<s>', '</s>']
    
    print("开始训练BPE分词器...")
    print("Starting BPE tokenizer training...")
    tokenizer.train(
        input_path=train_file,
        vocab_size=100,  # 减少词汇表大小以便快速测试 / Reduce vocabulary size for quick testing
        special_tokens=special_tokens,
        max_merges=20    # 减少合并次数 / Reduce merge operations
    )
    
    print(f"词汇表大小: {len(tokenizer.vocab)}")
    print(f"Vocabulary size: {len(tokenizer.vocab)}")
    print(f"合并操作数量: {len(tokenizer.merges)}")
    print(f"Number of merge operations: {len(tokenizer.merges)}")
    
    # 测试编码解码 / Test encoding and decoding
    test_text = "Hello world!"
    print(f"\n测试文本: '{test_text}'")
    print(f"Test text: '{test_text}'")
    
    token_ids = tokenizer.encode(test_text)
    print(f"编码结果: {token_ids}")
    print(f"Encoding result: {token_ids}")
    
    decoded_text = tokenizer.decode(token_ids)
    print(f"解码结果: '{decoded_text}'")
    print(f"Decoding result: '{decoded_text}'")
    
    # 清理临时文件 / Clean up temporary files
    if os.path.exists(train_file):
        os.remove(train_file)
    
    return tokenizer


if __name__ == "__main__":
    test_bpe_tokenizer()
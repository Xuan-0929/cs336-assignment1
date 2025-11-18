#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
并行BPE训练模块 / Parallel BPE Training Module
实现多进程并行处理大数据集的BPE分词器训练
Implements multi-process parallel processing for BPE tokenizer training on large datasets
"""

import os
import collections
import multiprocessing as mp
from typing import Dict, List, Tuple, BinaryIO
import heapq
from bpe_tokenizer import train_bpe


def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    将文件分块为可以独立处理的部分 / Split file into chunks that can be processed independently
    
    Args:
        file: 二进制文件对象 / Binary file object
        desired_num_chunks: 期望的块数 / Desired number of chunks
        split_special_token: 分割标记（字节格式）/ Split marker (byte format)
    
    Returns:
        分块边界位置列表 / List of chunk boundary positions
    """
    file_size = file.seek(0, 2)  # 获取文件大小 / Get file size
    file.seek(0)
    
    # 计算每个块的理想大小 / Calculate ideal size per chunk
    ideal_chunk_size = file_size // desired_num_chunks
    
    boundaries = [0]  # 起始边界 / Start boundary
    
    for i in range(1, desired_num_chunks):
        # 计算目标位置 / Calculate target position
        target_pos = i * ideal_chunk_size
        
        # 寻找合适的分割点 / Find suitable split point
        file.seek(target_pos)
        chunk = file.read(ideal_chunk_size)
        
        # 寻找分割标记或换行符 / Look for split marker or newline
        split_pos = -1
        
        # 优先寻找分割标记 / Priority to find split marker
        if split_special_token:
            split_pos = chunk.find(split_special_token)
        
        # 如果没有找到分割标记，寻找换行符 / If no split marker found, look for newline
        if split_pos == -1:
            split_pos = chunk.find(b'\n')
        
        # 如果还是没有找到，使用近似位置 / If still not found, use approximate position
        if split_pos == -1:
            split_pos = ideal_chunk_size // 2
        
        # 计算绝对位置 / Calculate absolute position
        absolute_pos = target_pos + split_pos + 1
        boundaries.append(min(absolute_pos, file_size))
    
    boundaries.append(file_size)  # 结束边界 / End boundary
    return boundaries


def process_chunk(args):
    """处理单个分块的worker函数 / Worker function to process individual chunks"""
    start, end, input_path, chunk_token, special_tokens = args
    
    word_freqs = collections.Counter()
    
    with open(input_path, 'rb') as f:
        f.seek(start)
        chunk_data = f.read(end - start)
        
        try:
            # 解码并处理分块 / Decode and process chunk
            text = chunk_data.decode('utf-8', errors='ignore')
            words = text.split()
            
            # 统计词频（转换为字节序列）/ Count word frequencies (convert to byte sequences)
            for word in words:
                if word:  # 跳过空词 / Skip empty words
                    byte_word = word.encode('utf-8')
                    word_freqs[byte_word] += 1
                    
        except Exception as e:
            print(f"处理分块 [{start}-{end}] 时出错: {e}")
            print(f"Error processing chunk [{start}-{end}]: {e}")
    
    return dict(word_freqs)


def process_chunks_parallel(input_path, num_processes, chunk_token, special_tokens):
    """并行处理所有分块 / Process all chunks in parallel"""
    
    # 获取分块边界 / Get chunk boundaries
    with open(input_path, 'rb') as f:
        boundaries = find_chunk_boundaries(f, num_processes, chunk_token)
    
    # 准备参数 / Prepare arguments
    chunk_args = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        chunk_args.append((start, end, input_path, chunk_token, special_tokens))
    
    # 并行处理 / Process in parallel
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(process_chunk, chunk_args)
    
    return results


def merge_chunk_statistics(chunk_results):
    """合并所有分块的统计结果 / Merge statistics from all chunks"""
    global_word_freqs = collections.Counter()
    
    for chunk_freqs in chunk_results:
        for word_bytes, freq in chunk_freqs.items():
            global_word_freqs[word_bytes] += freq
    
    return global_word_freqs


def train_bpe_from_statistics(
    word_freqs: Dict[bytes, int],
    vocab_size: int,
    special_tokens: List[str],
    max_merges: int = None
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    基于词频统计训练BPE / Train BPE based on word frequency statistics
    
    Args:
        word_freqs: 词频统计字典 / Word frequency statistics dictionary
        vocab_size: 目标词汇表大小 / Target vocabulary size
        special_tokens: 特殊标记列表 / Special tokens list
        max_merges: 最大合并次数 / Maximum number of merges
    
    Returns:
        vocab: 词汇表 / Vocabulary
        merges: 合并操作列表 / Merge operations list
    """
    
    # 初始化词汇表 / Initialize vocabulary
    vocab = {}
    next_id = 0
    
    # 添加基础字节词汇 / Add basic byte vocabulary
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
    
    # 构建初始词汇统计 / Build initial vocabulary statistics
    vocab_stats = {}
    for word_bytes, freq in word_freqs.items():
        tokens = [bytes([b]) for b in word_bytes]
        vocab_stats[tuple(tokens)] = freq
    
    # BPE训练循环 / BPE training loop
    merges = []
    merge_count = 0
    
    while len(vocab) < vocab_size:
        if max_merges is not None and merge_count >= max_merges:
            break
            
        # 统计字节对频率 / Count byte pair frequencies
        pair_freqs = collections.Counter()
        
        for tokens, freq in vocab_stats.items():
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pair_freqs[pair] += freq
        
        if not pair_freqs:
            break
        
        # 找到最频繁的字节对 / Find most frequent byte pair
        max_freq = max(pair_freqs.values())
        best_pairs = [pair for pair, freq in pair_freqs.items() if freq == max_freq]
        best_pair = best_pairs[-1] if best_pairs else None
        
        if best_pair is None:
            break
        
        if len(vocab) >= vocab_size:
            break
        
        # 创建新token / Create new token
        new_token = best_pair[0] + best_pair[1]
        vocab[next_id] = new_token
        next_id += 1
        
        # 记录合并 / Record merge
        merges.append(best_pair)
        merge_count += 1
        
        # 更新词汇统计 / Update vocabulary statistics
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


def parallel_bpe_training(
    input_path: str,
    vocab_size: int,
    special_tokens: List[str],
    max_merges: int = None,
    num_processes: int = 4,
    chunk_token: bytes = b"<|endoftext|>"
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    并行BPE训练主函数 / Parallel BPE training main function
    
    Args:
        input_path: 输入文件路径 / Input file path
        vocab_size: 目标词汇表大小 / Target vocabulary size
        special_tokens: 特殊标记列表 / Special tokens list
        max_merges: 最大合并次数 / Maximum number of merges
        num_processes: 进程数 / Number of processes
        chunk_token: 分块标记 / Chunk marker
    
    Returns:
        vocab: 词汇表 / Vocabulary
        merges: 合并操作列表 / Merge operations list
    """
    
    # 1. 分块处理 / Chunk processing
    chunk_results = process_chunks_parallel(
        input_path, num_processes, chunk_token, special_tokens
    )
    
    # 2. 合并统计结果 / Merge statistics
    global_word_freqs = merge_chunk_statistics(chunk_results)
    
    # 3. 全局BPE训练 / Global BPE training
    vocab, merges = train_bpe_from_statistics(
        global_word_freqs, vocab_size, special_tokens, max_merges
    )
    
    return vocab, merges


def create_large_sample_data(filename: str = "large_sample.txt", num_lines: int = 10000):
    """创建大样本训练数据 / Create large sample training data"""
    
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming the world.",
        "Natural language processing enables computers to understand text.",
        "Deep learning models require large amounts of training data.",
        "Tokenization is the first step in text processing.",
        "Byte Pair Encoding creates subword units for better representation.",
        "Parallel processing speeds up training on large datasets.",
        "Unicode enables text processing in multiple languages.",
        "UTF-8 encoding is efficient for multilingual text.",
        "Artificial intelligence is revolutionizing technology."
    ]
    
    with open(filename, 'w', encoding='utf-8') as f:
        for i in range(num_lines):
            text = sample_texts[i % len(sample_texts)]
            f.write(f"{text} This is line {i+1}.\n")
    
    return filename


def test_parallel_bpe():
    """测试并行BPE训练 / Test parallel BPE training"""
    print("=== 测试并行BPE训练 ===")
    print("=== Testing Parallel BPE Training ===")
    
    # 创建大样本数据 / Create large sample data
    train_file = create_large_sample_data(num_lines=5000)
    
    print(f"训练文件: {train_file}")
    print(f"Training file: {train_file}")
    print(f"文件大小: {os.path.getsize(train_file)} bytes")
    print(f"File size: {os.path.getsize(train_file)} bytes")
    
    # 并行训练参数 / Parallel training parameters
    vocab_size = 500
    special_tokens = ['<pad>', '<unk>', '<s>', '</s>', '<|endoftext|>']
    max_merges = 100
    num_processes = min(4, mp.cpu_count())
    
    print(f"进程数: {num_processes}")
    print(f"Number of processes: {num_processes}")
    print(f"词汇表大小: {vocab_size}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"最大合并次数: {max_merges}")
    print(f"Maximum merges: {max_merges}")
    
    # 开始并行训练 / Start parallel training
    print("\n开始并行BPE训练...")
    print("Starting parallel BPE training...")
    vocab, merges = parallel_bpe_training(
        input_path=train_file,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        max_merges=max_merges,
        num_processes=num_processes,
        chunk_token=b'<|endoftext|>'
    )
    
    print(f"训练完成!")
    print(f"Training completed!")
    print(f"最终词汇表大小: {len(vocab)}")
    print(f"Final vocabulary size: {len(vocab)}")
    print(f"实际合并次数: {len(merges)}")
    print(f"Actual merge operations: {len(merges)}")
    
    # 显示统计信息 / Display statistics
    print(f"\n词汇表统计:")
    print(f"Vocabulary statistics:")
    special_token_count = sum(1 for token in vocab.values() if token.startswith(b'<'))
    byte_token_count = sum(1 for token in vocab.values() if len(token) == 1)
    merge_token_count = len(vocab) - special_token_count - byte_token_count
    
    print(f"  特殊token: {special_token_count}")
    print(f"  Special tokens: {special_token_count}")
    print(f"  字节token: {byte_token_count}")
    print(f"  Byte tokens: {byte_token_count}")
    print(f"  合并token: {merge_token_count}")
    print(f"  Merge tokens: {merge_token_count}")
    
    # 显示前10个合并操作 / Display first 10 merge operations
    print(f"\n前10个合并操作:")
    print(f"First 10 merge operations:")
    for i, (token1, token2) in enumerate(merges[:10]):
        merged = token1 + token2
        print(f"  {i+1}: {repr(token1)} + {repr(token2)} -> {repr(merged)}")
    
    # 清理临时文件 / Clean up temporary files
    if os.path.exists(train_file):
        os.remove(train_file)
    
    return vocab, merges


if __name__ == "__main__":
    test_parallel_bpe()
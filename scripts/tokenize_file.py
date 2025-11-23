#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
便捷文件分词工具 / Convenient File Tokenization Tool
使用训练好的BPE分词器对指定文件进行分词处理
Tokenizes a specified file using the trained BPE tokenizer
"""

import argparse
import sys
import os
from typing import List, Tuple

# 获取项目根目录 / Get project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))
sys.path.insert(0, os.path.join(project_root, 'src', 'tokenizers'))

# 导入BPE分词器 / Import BPE tokenizer
from bpe_tokenizer import BPETokenizer
from utils import validate_file_path, format_file_size


def tokenize_file(input_file: str, vocab_file: str = None, merges_file: str = None, 
                  chunk_size: int = 1000, max_lines: int = None, 
                  output_format: str = "tokens") -> None:
    """
    对指定文件进行分词处理
    Tokenize the specified file
    
    Args:
        input_file: 输入文件路径 / Input file path
        vocab_file: 词汇表文件路径，默认为项目根目录的vocab.json / Vocabulary file path, default to vocab.json in project root
        merges_file: 合并规则文件路径，默认为项目根目录的merges.txt / Merge rules file path, default to merges.txt in project root
        chunk_size: 每次处理的文本块大小 / Text chunk size for processing
        max_lines: 最大处理行数，None表示处理全部 / Maximum number of lines to process, None for all
        output_format: 输出格式，可以是'tokens'（显示token IDs）、'text'（显示解码后的文本）或'both'（同时显示）
    """
    # 验证输入文件 / Validate input file
    if not validate_file_path(input_file):
        print(f"错误: 无法访问输入文件 '{input_file}'")
        print(f"Error: Cannot access input file '{input_file}'")
        return
    
    # 如果未指定词汇表和合并规则文件，则使用默认路径 / Use default paths if not specified
    if vocab_file is None:
        vocab_file = os.path.join(project_root, 'vocab.json')
    if merges_file is None:
        merges_file = os.path.join(project_root, 'merges.txt')
    
    # 验证词汇表和合并规则文件 / Validate vocab and merges files
    if not validate_file_path(vocab_file):
        print(f"错误: 无法访问词汇表文件 '{vocab_file}'")
        print(f"Error: Cannot access vocabulary file '{vocab_file}'")
        return
    
    if not validate_file_path(merges_file):
        print(f"错误: 无法访问合并规则文件 '{merges_file}'")
        print(f"Error: Cannot access merge rules file '{merges_file}'")
        return
    
    # 初始化分词器并加载模型 / Initialize tokenizer and load model
    print(f"加载分词器模型...")
    print(f"Loading tokenizer model...")
    tokenizer = BPETokenizer()
    try:
        tokenizer.load(vocab_file, merges_file)
        print(f"成功加载词汇表大小: {len(tokenizer.vocab)}, 合并规则数量: {len(tokenizer.merges)}")
        print(f"Successfully loaded vocabulary size: {len(tokenizer.vocab)}, merge rules count: {len(tokenizer.merges)}")
    except Exception as e:
        print(f"加载模型失败: {e}")
        print(f"Failed to load model: {e}")
        return
    
    # 处理文件 / Process file
    print(f"\n开始处理文件: {input_file}")
    print(f"Starting to process file: {input_file}")
    file_size = os.path.getsize(input_file)
    print(f"文件大小: {format_file_size(file_size)}")
    print(f"File size: {format_file_size(file_size)}")
    
    total_tokens = 0
    total_lines = 0
    total_chars = 0
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            line_number = 0  # 文件中的行号
            processed_count = 0  # 已处理的非空行数量
            
            for line in f:
                line_number += 1
                
                # 处理空行 / Skip empty lines
                stripped_line = line.strip()
                if not stripped_line:
                    continue
                
                processed_count += 1
                
                # 检查是否达到最大处理行数 / Check if max lines reached
                if max_lines and processed_count > max_lines:
                    break
                
                total_lines += 1
                total_chars += len(stripped_line)
                
                # 分词 / Tokenize
                tokens = tokenizer.encode(stripped_line)
                total_tokens += len(tokens)
                
                print(f"\n--- 第 {line_number} 行 (处理序号: {processed_count}) ---")
                print(f"--- Line {line_number} (Processed: {processed_count}) ---")
                
                # 显示原始文本片段 / Show original text snippet
                if len(stripped_line) > 60:
                    print(f"原始文本: {stripped_line[:30]}...{stripped_line[-30:]}")
                    print(f"Original text: {stripped_line[:30]}...{stripped_line[-30:]}")
                else:
                    print(f"原始文本: {stripped_line}")
                    print(f"Original text: {stripped_line}")
                
                # 显示结果 / Display results based on output format
                if output_format in ['tokens', 'both']:
                    # 对于长行，只显示部分token / Show partial tokens for long lines
                    if len(tokens) > 20:
                        print(f"Token IDs: {tokens[:10]} ... {tokens[-10:]} (共 {len(tokens)} 个tokens)")
                        print(f"Token IDs: {tokens[:10]} ... {tokens[-10:]} (total {len(tokens)} tokens)")
                    else:
                        print(f"Token IDs: {tokens}")
                        print(f"Token IDs: {tokens}")
                
                if output_format in ['text', 'both']:
                    # 解码并显示 / Decode and display
                    decoded = tokenizer.decode(tokens)
                    # 对于长行，只显示部分文本 / Show partial text for long lines
                    if len(decoded) > 60:
                        print(f"解码文本: {decoded[:30]}...{decoded[-30:]}")
                        print(f"Decoded text: {decoded[:30]}...{decoded[-30:]}")
                    else:
                        print(f"解码文本: {decoded}")
                        print(f"Decoded text: {decoded}")
                
                # 显示进度 / Show progress
                if processed_count % 10 == 0:
                    print(f"\n已处理 {processed_count} 行文本...")
                    print(f"Processed {processed_count} lines of text...")
                
    
    except Exception as e:
        print(f"处理文件时出错: {e}")
        print(f"Error processing file: {e}")
        return
    
    # 显示统计信息 / Show statistics
    print(f"=== 分词完成 ===")
    print(f"=== Tokenization Completed ===")
    print(f"处理行数: {total_lines}")
    print(f"Lines processed: {total_lines}")
    print(f"总字符数: {total_chars}")
    print(f"Total characters: {total_chars}")
    print(f"总token数: {total_tokens}")
    print(f"Total tokens: {total_tokens}")
    if total_chars > 0:
        compression_ratio = total_chars / total_tokens
        print(f"压缩率: {compression_ratio:.2f} (字符数/Token数)")
        print(f"Compression ratio: {compression_ratio:.2f} (characters/tokens)")
    

def main():
    """主函数 / Main function"""
    parser = argparse.ArgumentParser(
        description="便捷文件分词工具 - 使用BPE分词器对指定文件进行分词 / Convenient File Tokenization Tool - Tokenize files using BPE tokenizer"
    )
    
    # 必需参数 / Required arguments
    parser.add_argument('-i', '--input', required=True, help='要分词的输入文件路径 / Input file path to tokenize')
    
    # 可选参数 / Optional arguments
    parser.add_argument('-v', '--vocab', help='词汇表文件路径 (默认: vocab.json) / Vocabulary file path (default: vocab.json)')
    parser.add_argument('-m', '--merges', help='合并规则文件路径 (默认: merges.txt) / Merge rules file path (default: merges.txt)')
    parser.add_argument('-l', '--lines', type=int, help='最大处理行数 / Maximum number of lines to process')
    parser.add_argument('-f', '--format', default='tokens', choices=['tokens', 'text', 'both'],
                       help='输出格式 (默认: tokens) / Output format (default: tokens)')
    
    # 解析参数 / Parse arguments
    args = parser.parse_args()
    
    # 调用分词函数 / Call tokenization function
    tokenize_file(
        input_file=args.input,
        vocab_file=args.vocab,
        merges_file=args.merges,
        max_lines=args.lines,
        output_format=args.format
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BPE分词器项目主程序 / BPE Tokenizer Project Main Program
整合所有模块，提供统一的命令行界面
Integrates all modules and provides a unified command-line interface
"""

import argparse
import sys
import os
from typing import List

# 导入各个模块 / Import all modules
# 获取项目根目录 / Get project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))
sys.path.insert(0, os.path.join(project_root, 'src', 'tokenizers'))
sys.path.insert(0, os.path.join(project_root, 'src', 'encoders'))
sys.path.insert(0, os.path.join(project_root, 'src', 'optimizers'))
sys.path.insert(0, os.path.join(project_root, 'docs'))

from unicode_demo import demo_unicode_characters, analyze_unicode_encoding
from utf8_encoding import (
    compare_encoding_formats, 
    demonstrate_utf8_byte_patterns,
    demonstrate_encoding_decoding,
    explain_utf8_advantages
)
from bpe_tokenizer import BPETokenizer, create_sample_training_data, test_bpe_tokenizer
from parallel_bpe import parallel_bpe_training, create_large_sample_data
from dl_optimizers import test_all_optimizers
from utils import (
    setup_logging, 
    print_system_info, 
    validate_file_path,
    analyze_text_file,
    format_file_size
)


def demo_unicode(args):
    """Unicode演示功能 / Unicode demonstration function"""
    print("=== Unicode字符处理演示 ===")
    print("=== Unicode Character Processing Demo ===")
    
    demo_unicode_characters()
    analyze_unicode_encoding()
    
    if args.output:
        print(f"\n演示完成。输出已显示在屏幕上。")
        print(f"Demo completed. Output displayed on screen.")


def demo_utf8(args):
    """UTF-8编码演示功能 / UTF-8 encoding demonstration function"""
    print("=== UTF-8编码演示 ===")
    print("=== UTF-8 Encoding Demo ===")
    
    compare_encoding_formats()
    demonstrate_utf8_byte_patterns()
    demonstrate_encoding_decoding()
    explain_utf8_advantages()
    
    if args.output:
        print(f"\n演示完成。输出已显示在屏幕上。")
        print(f"Demo completed. Output displayed on screen.")


def train_bpe_model(args):
    """训练BPE模型 / Train BPE model"""
    print("=== 训练BPE分词器 ===")
    print("=== Training BPE Tokenizer ===")
    
    # 验证输入文件 / Validate input file
    if not validate_file_path(args.input):
        return
    
    # 设置特殊token / Set up special tokens
    special_tokens = ['<pad>', '<unk>', '<s>', '</s>']
    if args.special_tokens:
        special_tokens.extend(args.special_tokens.split(','))
    
    print(f"输入文件: {args.input}")
    print(f"Input file: {args.input}")
    print(f"词汇表大小: {args.vocab_size}")
    print(f"Vocabulary size: {args.vocab_size}")
    print(f"最大合并次数: {args.max_merges}")
    print(f"Maximum merges: {args.max_merges}")
    print(f"特殊token: {special_tokens}")
    print(f"Special tokens: {special_tokens}")
    
    # 根据文件大小选择训练方式 / Choose training method based on file size
    file_size = os.path.getsize(args.input)
    print(f"输入文件大小: {format_file_size(file_size)}")
    print(f"Input file size: {format_file_size(file_size)}")
    
    if file_size > 10 * 1024 * 1024:  # 大于10MB使用并行训练 / Use parallel training for files > 10MB
        print("文件较大，使用并行训练...")
        print("Large file, using parallel training...")
        vocab, merges = parallel_bpe_training(
            input_path=args.input,
            vocab_size=args.vocab_size,
            special_tokens=special_tokens,
            max_merges=args.max_merges,
            num_processes=args.num_processes
        )
    else:
        print("文件较小，使用单进程训练...")
        print("Small file, using single-process training...")
        tokenizer = BPETokenizer()
        tokenizer.train(
            input_path=args.input,
            vocab_size=args.vocab_size,
            special_tokens=special_tokens,
            max_merges=args.max_merges
        )
        vocab, merges = tokenizer.vocab, tokenizer.merges
    
    # 保存结果 / Save results
    if args.vocab_output:
        save_vocab(vocab, args.vocab_output)
        print(f"词汇表已保存到: {args.vocab_output}")
        print(f"Vocabulary saved to: {args.vocab_output}")
    
    if args.merges_output:
        save_merges(merges, args.merges_output)
        print(f"合并规则已保存到: {args.merges_output}")
        print(f"Merge rules saved to: {args.merges_output}")
    
    print(f"训练完成！词汇表大小: {len(vocab)}, 合并次数: {len(merges)}")
    print(f"Training completed! Vocabulary size: {len(vocab)}, merge operations: {len(merges)}")


def test_tokenizer(args):
    """测试分词器功能 / Test tokenizer functionality"""
    print("=== 测试BPE分词器 ===")
    print("=== Testing BPE Tokenizer ===")
    
    tokenizer = test_bpe_tokenizer()
    
    # 使用固定文本进行一次性测试 / Use fixed text for one-time testing
    text = "Hello world! 这是一个测试文本。"
    print(f"\n测试文本: '{text}'")
    print(f"Test text: '{text}'")
    
    try:
        token_ids = tokenizer.encode(text)
        print(f"Token IDs: {token_ids}")
        
        decoded = tokenizer.decode(token_ids)
        print(f"解码结果: '{decoded}'")
        print(f"Decoding result: '{decoded}'")
        
        print(f"Token数量: {len(token_ids)}")
        print(f"Token count: {len(token_ids)}")
        
        # 验证编码解码一致性 / Verify encoding-decoding consistency
        if text == decoded:
            print("\n✅ 编码解码一致性检查通过！")
            print("✅ Encoding-decoding consistency check passed!")
        else:
            print("\n❌ 编码解码一致性检查失败！")
            print("❌ Encoding-decoding consistency check failed!")
    except Exception as e:
        print(f"错误: {e}")
        print(f"Error: {e}")


def analyze_corpus(args):
    """分析语料库 / Analyze corpus"""
    print("=== 语料库分析 ===")
    print("=== Corpus Analysis ===")
    
    if not validate_file_path(args.input):
        return
    
    analysis = analyze_text_file(args.input)
    
    print(f"文件: {args.input}")
    print(f"File: {args.input}")
    print(f"大小: {format_file_size(analysis.get('total_characters', 0))}")
    print(f"Size: {format_file_size(analysis.get('total_characters', 0))}")
    print(f"行数: {analysis.get('total_lines', 0):,}")
    print(f"Lines: {analysis.get('total_lines', 0):,}")
    print(f"单词数: {analysis.get('total_words', 0):,}")
    print(f"Words: {analysis.get('total_words', 0):,}")
    print(f"唯一单词: {analysis.get('unique_words', 0):,}")
    print(f"Unique words: {analysis.get('unique_words', 0):,}")
    print(f"ASCII字符: {analysis.get('ascii_chars', 0):,}")
    print(f"ASCII characters: {analysis.get('ascii_chars', 0):,}")
    print(f"中文字符: {analysis.get('chinese_chars', 0):,}")
    print(f"Chinese characters: {analysis.get('chinese_chars', 0):,}")
    print(f"其他Unicode字符: {analysis.get('other_unicode_chars', 0):,}")
    print(f"Other Unicode characters: {analysis.get('other_unicode_chars', 0):,}")
    
    if args.output:
        import json
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        print(f"\n分析结果已保存到: {args.output}")
        print(f"Analysis results saved to: {args.output}")


def create_demo_data(args):
    """创建演示数据 / Create demo data"""
    print("=== 创建演示数据 ===")
    print("=== Creating Demo Data ===")
    
    if args.size == "small":
        num_lines = 1000
    elif args.size == "medium":
        num_lines = 10000
    elif args.size == "large":
        num_lines = 100000
    else:
        num_lines = int(args.size)
    
    output_file = args.output or f"demo_data_{num_lines}.txt"
    
    if args.parallel:
        create_large_sample_data(output_file, num_lines)
    else:
        create_sample_training_data()
        # 重命名文件 / Rename file
        if os.path.exists("sample_training_data.txt"):
            os.rename("sample_training_data.txt", output_file)
    
    file_size = os.path.getsize(output_file)
    print(f"演示数据已创建: {output_file}")
    print(f"Demo data created: {output_file}")
    print(f"文件大小: {format_file_size(file_size)}")
    print(f"File size: {format_file_size(file_size)}")
    print(f"行数: {num_lines:,}")
    print(f"Lines: {num_lines:,}")


def test_dl_optimizers(args):
    """测试深度学习优化器 / Test deep learning optimizers"""
    print("=== 测试深度学习优化器 ===")
    print("=== Testing Deep Learning Optimizers ===")
    
    test_all_optimizers()


def save_vocab(vocab, output_path: str):
    """保存词汇表 / Save vocabulary"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for token_id, token_bytes in vocab.items():
            token_str = token_bytes.decode('utf-8', errors='replace')
            f.write(f"{token_id}\t{repr(token_str)}\n")


def save_merges(merges, output_path: str):
    """保存合并规则 / Save merge rules"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, (token1, token2) in enumerate(merges):
            token1_str = token1.decode('utf-8', errors='replace')
            token2_str = token2.decode('utf-8', errors='replace')
            f.write(f"{i}\t{repr(token1_str)}\t{repr(token2_str)}\n")


def main():
    """主函数 / Main function"""
    parser = argparse.ArgumentParser(
        description="BPE分词器工具 - 支持Unicode处理、UTF-8编码演示和BPE训练 / BPE Tokenizer Tool - Supports Unicode processing, UTF-8 encoding demo, and BPE training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法 / Example usage:
  # Unicode演示 / Unicode demo
  python main.py demo-unicode
  
  # UTF-8编码演示 / UTF-8 encoding demo  
  python main.py demo-utf8
  
  # 训练BPE模型 / Train BPE model
  python main.py train -i data.txt -v 300 -o vocab.txt -m merges.txt
  
  # 测试分词器 / Test tokenizer
  python main.py test
  
  # 分析语料库 / Analyze corpus
  python main.py analyze -i data.txt -o analysis.json
  
  # 创建演示数据 / Create demo data
  python main.py create-data --size large -o big_data.txt
  
  # 测试深度学习优化器 / Test deep learning optimizers
  python main.py test-dl
        """
    )
    
    # 全局参数 / Global parameters
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='日志级别 / Log level')
    
    # 子命令 / Subcommands
    subparsers = parser.add_subparsers(dest='command', help='可用命令 / Available commands')
    
    # Unicode演示 / Unicode demo
    unicode_parser = subparsers.add_parser('demo-unicode', help='Unicode字符处理演示 / Unicode character processing demo')
    unicode_parser.add_argument('-o', '--output', help='输出文件（可选）/ Output file (optional)')
    
    # UTF-8演示 / UTF-8 demo
    utf8_parser = subparsers.add_parser('demo-utf8', help='UTF-8编码演示 / UTF-8 encoding demo')
    utf8_parser.add_argument('-o', '--output', help='输出文件（可选）/ Output file (optional)')
    
    # 训练BPE模型 / Train BPE model
    train_parser = subparsers.add_parser('train', help='训练BPE分词器 / Train BPE tokenizer')
    train_parser.add_argument('-i', '--input', required=True, help='输入训练文件 / Input training file')
    train_parser.add_argument('-v', '--vocab-size', type=int, default=300, 
                             help='词汇表大小（默认300）/ Vocabulary size (default 300)')
    train_parser.add_argument('-m', '--max-merges', type=int, default=100,
                             help='最大合并次数（默认100）/ Maximum merges (default 100)')
    train_parser.add_argument('--vocab-output', help='词汇表输出文件 / Vocabulary output file')
    train_parser.add_argument('--merges-output', help='合并规则输出文件 / Merge rules output file')
    train_parser.add_argument('--special-tokens', help='额外特殊token（逗号分隔）/ Additional special tokens (comma-separated)')
    train_parser.add_argument('-p', '--num-processes', type=int, 
                             default=min(4, os.cpu_count()),
                             help='进程数（默认自动检测）/ Number of processes (default auto-detect)')
    
    # 测试分词器 / Test tokenizer
    test_parser = subparsers.add_parser('test', help='测试BPE分词器 / Test BPE tokenizer')
    
    # 分析语料库 / Analyze corpus
    analyze_parser = subparsers.add_parser('analyze', help='分析语料库 / Analyze corpus')
    analyze_parser.add_argument('-i', '--input', required=True, help='输入文件 / Input file')
    analyze_parser.add_argument('-o', '--output', help='分析结果输出文件 / Analysis results output file')
    
    # 创建演示数据 / Create demo data
    data_parser = subparsers.add_parser('create-data', help='创建演示数据 / Create demo data')
    data_parser.add_argument('--size', default='medium', 
                            choices=['small', 'medium', 'large'],
                            help='数据大小 / Data size')
    data_parser.add_argument('-o', '--output', help='输出文件 / Output file')
    data_parser.add_argument('--parallel', action='store_true',
                            help='使用并行方式创建数据 / Use parallel method to create data')
    
    # 测试深度学习优化器 / Test deep learning optimizers
    dl_parser = subparsers.add_parser('test-dl', help='测试深度学习优化器 / Test deep learning optimizers')
    
    # 解析参数 / Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # 设置日志 / Set up logging
    setup_logging(args.log_level)
    
    # 执行相应命令 / Execute corresponding command
    if args.command == 'demo-unicode':
        demo_unicode(args)
    
    elif args.command == 'demo-utf8':
        demo_utf8(args)
    
    elif args.command == 'train':
        train_bpe_model(args)
    
    elif args.command == 'test':
        test_tokenizer(args)
    
    elif args.command == 'analyze':
        analyze_corpus(args)
    
    elif args.command == 'create-data':
        create_demo_data(args)
    
    elif args.command == 'test-dl':
        test_dl_optimizers(args)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
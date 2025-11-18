#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试所有模块的集成脚本
验证各个模块是否能正常工作
"""

import sys
import os
import subprocess
import tempfile


def test_module_import():
    """测试模块导入"""
    print("=== 测试模块导入 ===")
    
    modules = [
        'unicode_demo',
        'utf8_encoding', 
        'bpe_tokenizer',
        'parallel_bpe',
        'utils',
        'main'
    ]
    
    for module in modules:
        try:
            __import__(module)
            print(f"✓ {module}: 导入成功")
        except ImportError as e:
            print(f"✗ {module}: 导入失败 - {e}")
            return False
    
    return True


def test_unicode_demo():
    """测试Unicode演示模块"""
    print("\n=== 测试Unicode演示 ===")
    
    try:
        from unicode_demo import demo_unicode_characters, analyze_unicode_encoding
        
        # 捕获输出
        import io
        from contextlib import redirect_stdout
        
        f = io.StringIO()
        with redirect_stdout(f):
            demo_unicode_characters()
            analyze_unicode_encoding()
        
        output = f.getvalue()
        if "Unicode" in output and "chr(0)" in output:
            print("✓ Unicode演示: 运行成功")
            return True
        else:
            print("✗ Unicode演示: 输出验证失败")
            return False
            
    except Exception as e:
        print(f"✗ Unicode演示: 运行失败 - {e}")
        return False


def test_utf8_encoding():
    """测试UTF-8编码模块"""
    print("\n=== 测试UTF-8编码 ===")
    
    try:
        from utf8_encoding import (
            compare_encoding_formats,
            demonstrate_utf8_byte_patterns,
            demonstrate_encoding_decoding
        )
        
        # 测试核心函数
        compare_encoding_formats()
        demonstrate_utf8_byte_patterns()
        demonstrate_encoding_decoding()
        
        print("✓ UTF-8编码: 运行成功")
        return True
        
    except Exception as e:
        print(f"✗ UTF-8编码: 运行失败 - {e}")
        return False


def test_bpe_tokenizer():
    """测试BPE分词器"""
    print("\n=== 测试BPE分词器 ===")
    
    try:
        from bpe_tokenizer import BPETokenizer, create_sample_training_data
        
        # 创建临时训练数据
        train_file = create_sample_training_data()
        
        # 训练分词器
        tokenizer = BPETokenizer()
        tokenizer.train(
            input_path=train_file,
            vocab_size=100,
            special_tokens=['<pad>', '<unk>'],
            max_merges=20
        )
        
        # 测试编码解码
        test_text = "Hello world!"
        token_ids = tokenizer.encode(test_text)
        decoded_text = tokenizer.decode(token_ids)
        
        print(f"原始文本: {test_text}")
        print(f"Token IDs: {token_ids}")
        print(f"解码结果: {decoded_text}")
        print(f"词汇表大小: {len(tokenizer.vocab)}")
        
        # 清理
        if os.path.exists(train_file):
            os.remove(train_file)
        
        print("✓ BPE分词器: 运行成功")
        return True
        
    except Exception as e:
        print(f"✗ BPE分词器: 运行失败 - {e}")
        return False


def test_utils():
    """测试工具函数"""
    print("\n=== 测试工具函数 ===")
    
    try:
        from utils import (
            format_file_size, 
            create_sample_text_file,
            analyze_text_file,
            print_system_info
        )
        
        # 测试文件大小格式化
        size_tests = [
            (0, "0 B"),
            (1024, "1.00 KB"),
            (1048576, "1.00 MB"),
        ]
        
        for size, expected in size_tests:
            result = format_file_size(size)
            if expected in result:
                print(f"✓ 文件大小格式化: {size} -> {result}")
        
        # 创建测试文件
        test_file = "test_utils.txt"
        create_sample_text_file(test_file, num_lines=100)
        
        # 分析文件
        analysis = analyze_text_file(test_file)
        if analysis and "total_lines" in analysis:
            print(f"✓ 文件分析: 成功分析 {analysis['total_lines']} 行")
        
        # 系统信息
        print_system_info()
        
        # 清理
        if os.path.exists(test_file):
            os.remove(test_file)
        
        print("✓ 工具函数: 运行成功")
        return True
        
    except Exception as e:
        print(f"✗ 工具函数: 运行失败 - {e}")
        return False


def test_main_cli():
    """测试主程序CLI"""
    print("\n=== 测试主程序CLI ===")
    
    try:
        # 测试帮助信息
        result = subprocess.run([sys.executable, 'main.py', '--help'], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0 and "BPE分词器工具" in result.stdout:
            print("✓ 主程序CLI: 帮助信息正常")
        else:
            print("✗ 主程序CLI: 帮助信息异常")
            return False
        
        # 测试Unicode演示
        result = subprocess.run([sys.executable, 'main.py', 'demo-unicode'], 
                              capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✓ 主程序CLI: Unicode演示正常")
        else:
            print("✗ 主程序CLI: Unicode演示失败")
            return False
        
        return True
        
    except subprocess.TimeoutExpired:
        print("✗ 主程序CLI: 超时")
        return False
    except Exception as e:
        print(f"✗ 主程序CLI: 运行失败 - {e}")
        return False


def test_parallel_bpe():
    """测试并行BPE"""
    print("\n=== 测试并行BPE ===")
    
    try:
        from parallel_bpe import create_large_sample_data
        
        # 创建小测试数据
        test_file = "test_parallel.txt"
        create_large_sample_data(test_file, num_lines=1000)
        
        if os.path.exists(test_file):
            size = os.path.getsize(test_file)
            print(f"✓ 并行BPE: 成功创建测试文件 ({size} bytes)")
            
            # 清理
            os.remove(test_file)
            
            print("✓ 并行BPE: 运行成功")
            return True
        else:
            print("✗ 并行BPE: 文件创建失败")
            return False
            
    except Exception as e:
        print(f"✗ 并行BPE: 运行失败 - {e}")
        return False


def main():
    """主测试函数"""
    print("BPE分词器项目 - 模块集成测试")
    print("=" * 50)
    
    # 保存当前目录
    original_dir = os.getcwd()
    
    try:
        # 切换到输出目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if script_dir:
            os.chdir(script_dir)
        
        # 运行所有测试
        tests = [
            test_module_import,
            test_unicode_demo,
            test_utf8_encoding,
            test_bpe_tokenizer,
            test_utils,
            test_parallel_bpe,
            test_main_cli,
        ]
        
        passed = 0
        total = len(tests)
        
        for test in tests:
            if test():
                passed += 1
        
        # 总结
        print("\n" + "=" * 50)
        print(f"测试结果: {passed}/{total} 通过")
        
        if passed == total:
            print("🎉 所有测试通过！项目模块集成正常。")
            return 0
        else:
            print("❌ 部分测试失败，请检查错误信息。")
            return 1
            
    finally:
        # 恢复目录
        os.chdir(original_dir)


if __name__ == "__main__":
    sys.exit(main())
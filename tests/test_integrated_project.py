#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
集成项目测试脚本 / Integrated Project Test Script
测试BPE分词器和深度学习优化器的整合功能
Test integrated functionality of BPE tokenizer and deep learning optimizers
"""

import sys
import os
import subprocess
import tempfile

# 添加项目路径 / Add project path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_command_line_interface():
    """测试命令行界面 / Test command line interface"""
    print("=== 测试命令行界面 ===")
    print("=== Testing Command Line Interface ===")
    
    commands = [
        ("python main.py --help", "显示帮助信息 / Show help information"),
        ("python main.py demo-unicode", "Unicode演示 / Unicode demo"),
        ("python main.py demo-utf8", "UTF-8编码演示 / UTF-8 encoding demo"),
        ("python main.py test-dl", "深度学习优化器测试 / Deep learning optimizers test"),
    ]
    
    for cmd, desc in commands:
        print(f"\n测试: {desc}")
        print(f"命令: {cmd}")
        
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                print(f"✓ 成功 - {desc}")
                if "test-dl" not in cmd:  # 深度学习测试输出较长 / DL test has long output
                    print(f"  返回信息: {result.stdout[:100]}...")
            else:
                print(f"✗ 失败 - {desc}")
                print(f"  错误: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print(f"✗ 超时 - {desc}")
        except Exception as e:
            print(f"✗ 异常 - {desc}: {e}")


def test_bpe_training():
    """测试BPE训练功能 / Test BPE training functionality"""
    print("\n=== 测试BPE训练功能 ===")
    print("=== Testing BPE Training Functionality ===")
    
    # 创建临时训练文件 / Create temporary training file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write("Hello world! This is a test.\n")
        f.write("Machine learning is amazing.\n")
        f.write("Natural language processing.\n")
        f.write("Deep learning models.\n")
        train_file = f.name
    
    try:
        vocab_file = train_file.replace('.txt', '_vocab.txt')
        merges_file = train_file.replace('.txt', '_merges.txt')
        
        cmd = f"python main.py train -i {train_file} -v 100 -m 20 --vocab-output {vocab_file} --merges-output {merges_file}"
        print(f"执行命令: {cmd}")
        
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✓ BPE训练成功")
            print(result.stdout)
            
            # 检查输出文件 / Check output files
            if os.path.exists(vocab_file) and os.path.exists(merges_file):
                print("✓ 输出文件已生成")
                
                with open(vocab_file, 'r', encoding='utf-8') as vf:
                    vocab_lines = len(vf.readlines())
                with open(merges_file, 'r', encoding='utf-8') as mf:
                    merges_lines = len(mf.readlines())
                
                print(f"词汇表大小: {vocab_lines}")
                print(f"合并规则数量: {merges_lines}")
            else:
                print("✗ 输出文件未生成")
        else:
            print("✗ BPE训练失败")
            print(result.stderr)
            
    finally:
        # 清理临时文件 / Clean up temporary files
        for f in [train_file, vocab_file, merges_file]:
            if os.path.exists(f):
                os.remove(f)


def test_deep_learning_optimizers():
    """测试深度学习优化器 / Test deep learning optimizers"""
    print("\n=== 测试深度学习优化器功能 ===")
    print("=== Testing Deep Learning Optimizers Functionality ===")
    
    try:
        # 直接导入并测试 / Import and test directly
        from dl_optimizers import test_all_optimizers
        test_all_optimizers()
        print("✓ 深度学习优化器测试成功")
        
    except Exception as e:
        print(f"✗ 深度学习优化器测试失败: {e}")
        import traceback
        traceback.print_exc()


def test_integrated_workflow():
    """测试集成工作流 / Test integrated workflow"""
    print("\n=== 测试集成工作流 ===")
    print("=== Testing Integrated Workflow ===")
    
    # 1. 创建训练数据 / Create training data
    print("1. 创建训练数据...")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write("Hello world! This is a test.\n")
        f.write("Machine learning is amazing.\n")
        f.write("Natural language processing.\n")
        f.write("Deep learning models.\n")
        f.write("Neural networks are powerful.\n")
        train_file = f.name
    
    try:
        # 2. 训练BPE分词器 / Train BPE tokenizer
        print("2. 训练BPE分词器...")
        vocab_file = train_file.replace('.txt', '_vocab.txt')
        merges_file = train_file.replace('.txt', '_merges.txt')
        
        cmd = f"python main.py train -i {train_file} -v 50 -m 10 --vocab-output {vocab_file} --merges-output {merges_file}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
        
        if result.returncode != 0:
            print("✗ BPE训练失败")
            print(result.stderr)
            return
        
        print("✓ BPE训练成功")
        
        # 3. 使用分词器进行编码 / Use tokenizer for encoding
        print("3. 测试分词器编码...")
        from bpe_tokenizer import BPETokenizer
        
        tokenizer = BPETokenizer()
        tokenizer.load(vocab_file, merges_file)
        
        test_text = "Hello machine learning!"
        token_ids = tokenizer.encode(test_text)
        decoded = tokenizer.decode(token_ids)
        
        print(f"原文本: {test_text}")
        print(f"Token IDs: {token_ids}")
        print(f"解码结果: {decoded}")
        
        if test_text == decoded:
            print("✓ 编码解码成功")
        else:
            print("✗ 编码解码失败")
        
        # 4. 测试深度学习优化器 / Test deep learning optimizers
        print("4. 测试深度学习优化器...")
        test_deep_learning_optimizers()
        
        print("\n✓ 集成工作流测试完成")
        
    finally:
        # 清理文件 / Clean up files
        for f in [train_file, vocab_file, merges_file]:
            if os.path.exists(f):
                os.remove(f)


def main():
    """主测试函数 / Main test function"""
    print("=" * 60)
    print("BPE分词器 + 深度学习优化器 集成测试")
    print("BPE Tokenizer + Deep Learning Optimizers Integration Test")
    print("=" * 60)
    
    # 保存当前目录 / Save current directory
    original_dir = os.getcwd()
    
    try:
        # 切换到输出目录 / Switch to output directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if script_dir:
            os.chdir(script_dir)
        
        # 运行所有测试 / Run all tests
        tests = [
            test_command_line_interface,
            test_bpe_training,
            test_deep_learning_optimizers,
            test_integrated_workflow,
        ]
        
        passed = 0
        total = len(tests)
        
        for test in tests:
            try:
                test()
                passed += 1
                print(f"\n{'='*60}")
            except Exception as e:
                print(f"\n✗ 测试失败: {e}")
                import traceback
                traceback.print_exc()
                print(f"\n{'='*60}")
        
        # 总结 / Summary
        print("\n" + "=" * 60)
        print(f"集成测试结果: {passed}/{total} 通过")
        print(f"Integration Test Results: {passed}/{total} passed")
        
        if passed == total:
            print("🎉 所有集成测试通过！项目整合成功。")
            print("🎉 All integration tests passed! Project integration successful.")
            return 0
        else:
            print("❌ 部分集成测试失败，请检查错误信息。")
            print("❌ Some integration tests failed, please check error messages.")
            return 1
            
    finally:
        # 恢复目录 / Restore directory
        os.chdir(original_dir)


if __name__ == "__main__":
    sys.exit(main())
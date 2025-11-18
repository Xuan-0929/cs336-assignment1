#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UTF-8编码处理模块 / UTF-8 Encoding Processing Module
演示UTF-8编码原理，不同编码方式的比较，以及错误处理
Demonstrates UTF-8 encoding principles, comparison of different encoding methods, and error handling
"""

import sys


def compare_encoding_formats():
    """比较不同编码格式的效率和特点 / Compare efficiency and characteristics of different encoding formats"""
    print("=== 编码格式比较 ===")
    print("=== Encoding Format Comparison ===")
    
    test_strings = [
        "Hello World",  # 纯英文
        "Hello 世界",   # 混合
        "你好，世界",   # 纯中文
        "Café résumé",  # 欧洲语言
    ]
    
    for text in test_strings:
        print(f"\n文本: '{text}'")
        print(f"Text: '{text}'")
        
        # UTF-8编码 / UTF-8 encoding
        utf8_bytes = text.encode('utf-8')
        print(f"UTF-8:  {len(utf8_bytes)} bytes - {utf8_bytes}")
        
        # UTF-16编码 / UTF-16 encoding
        utf16_bytes = text.encode('utf-16')
        print(f"UTF-16: {len(utf16_bytes)} bytes - {utf16_bytes[:20]}...")
        
        # UTF-32编码 / UTF-32 encoding
        try:
            utf32_bytes = text.encode('utf-32')
            print(f"UTF-32: {len(utf32_bytes)} bytes - {utf32_bytes[:20]}...")
        except Exception as e:
            print(f"UTF-32编码错误: {e}")
            print(f"UTF-32 encoding error: {e}")


def demonstrate_utf8_byte_patterns():
    """演示UTF-8字节模式 / Demonstrate UTF-8 byte patterns"""
    print("\n=== UTF-8字节模式演示 ===")
    print("=== UTF-8 Byte Pattern Demo ===")
    
    # 单字节字符 (ASCII) / Single-byte character (ASCII)
    ascii_char = 'A'
    print(f"ASCII字符 '{ascii_char}': {ascii_char.encode('utf-8')} (1字节)")
    print(f"ASCII character '{ascii_char}': {ascii_char.encode('utf-8')} (1 byte)")
    
    # 双字节字符 / Two-byte character
    two_byte_char = 'é'
    print(f"双字节字符 '{two_byte_char}': {two_byte_char.encode('utf-8')} (2字节)")
    print(f"Two-byte character '{two_byte_char}': {two_byte_char.encode('utf-8')} (2 bytes)")
    
    # 三字节字符 / Three-byte character
    three_byte_char = '中'
    print(f"三字节字符 '{three_byte_char}': {three_byte_char.encode('utf-8')} (3字节)")
    print(f"Three-byte character '{three_byte_char}': {three_byte_char.encode('utf-8')} (3 bytes)")
    
    # 四字节字符 / Four-byte character
    four_byte_char = '😀'
    print(f"四字节字符 '{four_byte_char}': {four_byte_char.encode('utf-8')} (4字节)")
    print(f"Four-byte character '{four_byte_char}': {four_byte_char.encode('utf-8')} (4 bytes)")


def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    """
    错误的UTF-8解码方法 - 逐字节解码
    这个方法会失败，因为UTF-8字符可能是多字节的
    
    Wrong UTF-8 decoding method - byte-by-byte decoding
    This method will fail because UTF-8 characters may be multi-byte
    """
    try:
        return "".join([bytes([b]).decode("utf-8") for b in bytestring])
    except UnicodeDecodeError as e:
        print(f"解码错误: {e}")
        return None


def decode_utf8_bytes_to_str_correct(bytestring: bytes):
    """
    正确的UTF-8解码方法 - 整体解码
    
    Correct UTF-8 decoding method - holistic decoding
    """
    try:
        return bytestring.decode("utf-8")
    except UnicodeDecodeError as e:
        print(f"解码错误: {e}")
        return None


def demonstrate_encoding_decoding():
    """演示编码和解码过程 / Demonstrate encoding and decoding process"""
    print("\n=== 编码解码演示 ===")
    print("=== Encoding and Decoding Demo ===")
    
    # 测试字符串 / Test strings
    test_strings = ["hello", "中文", "Hello 世界", "🌟🎉"]
    
    for text in test_strings:
        print(f"\n原始文本: '{text}'")
        print(f"Original text: '{text}'")
        
        # 编码为UTF-8字节 / Encode to UTF-8 bytes
        encoded = text.encode('utf-8')
        print(f"UTF-8编码: {encoded}")
        print(f"UTF-8 encoding: {encoded}")
        
        # 正确解码 / Correct decoding
        decoded_correct = decode_utf8_bytes_to_str_correct(encoded)
        print(f"正确解码: '{decoded_correct}'")
        print(f"Correct decoding: '{decoded_correct}'")
        
        # 错误解码尝试 / Wrong decoding attempt
        if len(encoded) > 1:  # 只对多字节字符串测试错误方法 / Only test wrong method for multi-byte strings
            decoded_wrong = decode_utf8_bytes_to_str_wrong(encoded)
            print(f"错误解码: {decoded_wrong}")
            print(f"Wrong decoding: {decoded_wrong}")


def explain_utf8_advantages():
    """解释UTF-8相比其他编码的优势"""
    print("\n=== UTF-8编码优势 ===")
    
    advantages = [
        "1. 向后兼容ASCII - 所有ASCII字符在UTF-8中保持单字节表示",
        "2. 变长编码 - 根据字符复杂度使用1-4字节，节省空间",
        "3. 自同步 - 可以从字节流中任意位置开始解码",
        "4. 字节顺序无关 - 不需要BOM（字节顺序标记）",
        "5. 广泛支持 - 互联网上最常用的编码格式",
        "6. 适合Tokenizer - 为BPE等算法提供最佳起点"
    ]
    
    for advantage in advantages:
        print(advantage)
    
    print("\n=== 与其他编码比较 ===")
    
    comparisons = [
        "UTF-16: 大多数字符使用2字节，但英文文本效率低（大量00字节）",
        "UTF-32: 所有字符固定4字节，空间浪费严重",
        "GBK/GB2312: 仅支持中文，不适合多语言处理",
        "ISO-8859-1: 仅支持拉丁字符，无法处理中文等"
    ]
    
    for comparison in comparisons:
        print(comparison)


def show_invalid_utf8_sequences():
    """显示无效的UTF-8序列示例"""
    print("\n=== 无效UTF-8序列示例 ===")
    
    print("有效的UTF-8双字节序列格式: 110xxxxx 10xxxxxx")
    print("无效的序列示例:")
    
    # 无效的起始字节
    invalid_sequences = [
        b'\xC1\xBF',  # 无效起始字节
        b'\x80\x80',  # 无效起始字节
        b'\xFF\xFF',  # 无效字节
    ]
    
    for seq in invalid_sequences:
        try:
            decoded = seq.decode('utf-8')
            print(f"序列 {seq.hex()}: 意外解码成功: {repr(decoded)}")
        except UnicodeDecodeError as e:
            print(f"序列 {seq.hex()}: 解码失败 - {e}")


if __name__ == "__main__":
    compare_enco
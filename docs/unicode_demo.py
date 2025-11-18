#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unicode字符处理演示模块 / Unicode Character Processing Demo Module
演示ord()和chr()函数的使用，以及Unicode字符的基本操作
Demonstrates the usage of ord() and chr() functions, and basic Unicode character operations
"""


def demo_unicode_characters():
    """演示Unicode字符的基本操作 / Demonstrate basic Unicode character operations"""
    print("=== Unicode字符处理演示 ===")
    print("=== Unicode Character Processing Demo ===")
    
    # 演示chr(0) - 空字符
    print(f"chr(0): {repr(chr(0))}")
    print(f"chr(0)显示: {chr(0)}")
    
    # 演示ord()函数
    print(f"ord('A'): {ord('A')}")
    print(f"ord('中'): {ord('中')}")
    
    # 演示chr()函数
    print(f"chr(65): {chr(65)}")
    print(f"chr(20013): {chr(20013)}")
    
    # 演示空字符在字符串中的行为
    test_string = "this is a test" + chr(0) + "string"
    print(f"包含空字符的字符串: {test_string}")
    print(f"字符串长度: {len(test_string)}")
    
    # 演示repr()函数
    newline_char = '\n'
    print(f"repr('\\n'): {repr(newline_char)}")
    print(f"直接打印'\\n': {newline_char}")


def analyze_unicode_encoding():
    """分析Unicode编码 / Analyze Unicode encoding"""
    print("\n=== Unicode编码分析 ===")
    print("=== Unicode Encoding Analysis ===")
    
    # 英文字符 / English character
    english_char = 'a'
    print(f"字符 '{english_char}' 的Unicode码点: {ord(english_char)}")
    print(f"Character '{english_char}' Unicode code point: {ord(english_char)}")
    print(f"字符 '{english_char}' 的UTF-8编码: {english_char.encode('utf-8')}")
    print(f"Character '{english_char}' UTF-8 encoding: {english_char.encode('utf-8')}")
    
    # 中文字符 / Chinese character
    chinese_char = '中'
    print(f"字符 '{chinese_char}' 的Unicode码点: {ord(chinese_char)}")
    print(f"Character '{chinese_char}' Unicode code point: {ord(chinese_char)}")
    print(f"字符 '{chinese_char}' 的UTF-8编码: {chinese_char.encode('utf-8')}")
    print(f"Character '{chinese_char}' UTF-8 encoding: {chinese_char.encode('utf-8')}")
    
    # 特殊字符 / Special character
    special_char = 'é'
    print(f"字符 '{special_char}' 的Unicode码点: {ord(special_char)}")
    print(f"Character '{special_char}' Unicode code point: {ord(special_char)}")
    print(f"字符 '{special_char}' 的UTF-8编码: {special_char.encode('utf-8')}")
    print(f"Character '{special_char}' UTF-8 encoding: {special_char.encode('utf-8')}")


if __name__ == "__main__":
    demo_unicode_characters()
    analyze_unicode_encoding()
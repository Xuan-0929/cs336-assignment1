#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unicode Character Processing Demo Module
Demonstrates the usage of ord() and chr() functions, and basic Unicode character operations
"""


def demo_unicode_characters():
    """Demonstrate basic Unicode character operations"""
    print("=== Unicode Character Processing Demo ===")
    
    # Demonstrate chr(0) - null character
    print(f"chr(0): {repr(chr(0))}")
    print(f"chr(0) display: {chr(0)}")
    
    # Demonstrate ord() function
    print(f"ord('A'): {ord('A')}")
    print(f"ord('中'): {ord('中')}")
    
    # Demonstrate chr() function
    print(f"chr(65): {chr(65)}")
    print(f"chr(20013): {chr(20013)}")
    
    # Demonstrate null character behavior in strings
    test_string = "this is a test" + chr(0) + "string"
    print(f"String containing null character: {test_string}")
    print(f"String length: {len(test_string)}")
    
    # Demonstrate repr() function
    newline_char = '\n'
    print(f"repr('\\n'): {repr(newline_char)}")
    print(f"Direct print '\\n': {newline_char}")


def analyze_unicode_encoding():
    """Analyze Unicode encoding"""
    print("\n=== Unicode Encoding Analysis ===")
    
    # English character
    english_char = 'a'
    print(f"Character '{english_char}' Unicode code point: {ord(english_char)}")
    print(f"Character '{english_char}' UTF-8 encoding: {english_char.encode('utf-8')}")
    
    # Chinese character
    chinese_char = '中'
    print(f"Character '{chinese_char}' Unicode code point: {ord(chinese_char)}")
    print(f"Character '{chinese_char}' UTF-8 encoding: {chinese_char.encode('utf-8')}")
    
    # Special character
    special_char = 'é'
    print(f"Character '{special_char}' Unicode code point: {ord(special_char)}")
    print(f"Character '{special_char}' UTF-8 encoding: {special_char.encode('utf-8')}")


if __name__ == "__main__":
    demo_unicode_characters()
    analyze_unicode_encoding()
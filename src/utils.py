#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工具函数模块 / Utility Functions Module
包含各种辅助函数和通用工具
Contains various auxiliary functions and general utilities
"""

import os
import time
import logging
from typing import List, Dict, Any, Optional


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """设置日志配置 / Set up logging configuration"""
    
    # 创建日志格式
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # 配置日志
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(),
        ]
    )
    
    # 如果指定了日志文件，添加文件处理器
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)


def get_file_stats(file_path: str) -> Dict[str, Any]:
    """获取文件统计信息 / Get file statistics"""
    if not os.path.exists(file_path):
        return {}
    
    stats = os.stat(file_path)
    return {
        "size": stats.st_size,
        "created": time.ctime(stats.st_ctime),
        "modified": time.ctime(stats.st_mtime),
        "accessed": time.ctime(stats.st_atime),
        "is_file": os.path.isfile(file_path),
        "is_dir": os.path.isdir(file_path),
    }


def format_file_size(size_bytes: int) -> str:
    """格式化文件大小 / Format file size"""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.2f} {size_names[i]}"


def create_sample_text_file(filename: str, num_lines: int = 1000, languages: List[str] = None) -> str:
    """创建示例文本文件 / Create sample text file"""
    
    if languages is None:
        languages = ["english", "chinese", "mixed"]
    
    # 示例文本模板
    text_templates = {
        "english": [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning algorithms process large datasets.",
            "Natural language processing enables text understanding.",
            "Deep neural networks learn complex patterns.",
            "Artificial intelligence transforms industries.",
        ],
        "chinese": [
            "机器学习是人工智能的重要分支。",
            "自然语言处理让计算机理解人类语言。",
            "深度学习模型需要大量训练数据。",
            "分词是中文文本处理的关键步骤。",
            "神经网络可以学习复杂的语言模式。",
        ],
        "mixed": [
            "Hello 世界！Machine learning 真神奇。",
            "深度学习 deep learning 是当前热点。",
            "自然语言处理 NLP 让计算机更智能。",
            "人工智能 AI 正在改变我们的世界。",
            "Tokenizer 分词器是文本处理的基础。",
        ]
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        for i in range(num_lines):
            lang = languages[i % len(languages)]
            templates = text_templates.get(lang, text_templates["english"])
            text = templates[i % len(templates)]
            f.write(f"{text} Line {i+1}.\n")
    
    return filename


def benchmark_function(func, *args, **kwargs):
    """基准测试函数执行时间 / Benchmark function execution time"""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    execution_time = end_time - start_time
    
    return result, execution_time


def print_system_info():
    """打印系统信息 / Print system information"""
    import platform
    import sys
    
    print("=== 系统信息 ===")
    print("=== System Information ===")
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"Operating System: {platform.system()} {platform.release()}")
    print(f"Python版本: {sys.version}")
    print(f"Python Version: {sys.version}")
    print(f"CPU数量: {os.cpu_count()}")
    print(f"CPU Count: {os.cpu_count()}")
    print(f"编码: {sys.getdefaultencoding()}")
    print(f"Encoding: {sys.getdefaultencoding()}")


def validate_file_path(file_path: str, should_exist: bool = True) -> bool:
    """验证文件路径 / Validate file path"""
    if should_exist:
        if not os.path.exists(file_path):
            print(f"错误: 文件 '{file_path}' 不存在")
            return False
        if not os.path.isfile(file_path):
            print(f"错误: '{file_path}' 不是文件")
            return False
    else:
        # 检查目录是否存在且可写
        dir_path = os.path.dirname(file_path)
        if dir_path and not os.path.exists(dir_path):
            try:
                os.makedirs(dir_path, exist_ok=True)
            except OSError as e:
                print(f"错误: 无法创建目录 '{dir_path}': {e}")
                return False
    
    return True


def safe_remove_file(file_path: str) -> bool:
    """安全删除文件 / Safely remove file"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            return True
    except OSError as e:
        print(f"警告: 无法删除文件 '{file_path}': {e}")
    
    return False


def list_python_files(directory: str = ".") -> List[str]:
    """列出目录中的Python文件 / List Python files in directory"""
    python_files = []
    
    for item in os.listdir(directory):
        if item.endswith('.py') and os.path.isfile(os.path.join(directory, item)):
            python_files.append(item)
    
    return sorted(python_files)


def get_memory_usage():
    """获取当前内存使用情况 / Get current memory usage"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return {
            "rss": memory_info.rss,  # 常驻内存集
            "vms": memory_info.vms,  # 虚拟内存大小
            "rss_human": format_file_size(memory_info.rss),
            "vms_human": format_file_size(memory_info.vms),
        }
    except ImportError:
        return {"error": "psutil not available"}


def print_progress_bar(current: int, total: int, prefix: str = "", suffix: str = "", length: int = 50):
    """打印进度条 / Print progress bar"""
    percent = float(current) / total if total > 0 else 0
    filled_length = int(length * percent)
    bar = "█" * filled_length + "-" * (length - filled_length)
    
    print(f"\r{prefix} |{bar}| {percent:.1%} {suffix}", end="", flush=True)
    
    if current == total:
        print()  # 换行


def analyze_text_file(file_path: str) -> Dict[str, Any]:
    """分析文本文件内容 / Analyze text file content"""
    if not validate_file_path(file_path):
        return {}
    
    stats = get_file_stats(file_path)
    
    # 读取文件内容进行分析
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\n')
        words = content.split()
        
        analysis = {
            "total_characters": len(content),
            "total_lines": len(lines),
            "total_words": len(words),
            "unique_words": len(set(words)),
            "file_stats": stats,
        }
        
        # 字符类型统计
        ascii_count = sum(1 for c in content if ord(c) < 128)
        chinese_count = sum(1 for c in content if '\u4e00' <= c <= '\u9fff')
        
        analysis.update({
            "ascii_chars": ascii_count,
            "chinese_chars": chinese_count,
            "other_unicode_chars": len(content) - ascii_count - chinese_count,
        })
        
        return analysis
        
    except Exception as e:
        return {"error": f"无法分析文件: {e}"}


def save_results_to_file(data: Dict[str, Any], output_path: str, format_type: str = "json"):
    """将结果保存到文件 / Save results to file"""
    
    if format_type.lower() == "json":
        import json
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    elif format_type.lower() == "txt":
        with open(output_path, 'w', encoding='utf-8') as f:
            for key, value in data.items():
                f.write(f"{key}: {value}\n")
    
    else:
        raise ValueError(f"不支持的格式: {format_type}")


if __name__ == "__main__":
    # 测试工具函数
    print_system_info()
    
    # 创建示例文件
    sample_file = "test_utils.txt"
    create_sample_text_file(sample_file, num_lines=100)
    
    # 分析文件
    analysis = analyze_text_file(sample_file)
    print(f"\n文件分析结果:")
    for key, value in analysis.items():
        print(f"  {key}: {value}")
    
    # 清理
    safe_remove_file(sample_file)
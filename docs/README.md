# BPE分词器项目

这是一个基于字节对编码（Byte Pair Encoding, BPE）的分词器实现项目，将Jupyter Notebook中的多个功能模块拆分为独立的Python文件，提供更清晰的项目结构和更好的可维护性。

## 项目结构

```
├── main.py                 # 主程序入口，提供命令行界面
├── unicode_demo.py         # Unicode字符处理演示
├── utf8_encoding.py        # UTF-8编码处理和分析
├── bpe_tokenizer.py        # BPE分词器核心实现
├── parallel_bpe.py         # 并行BPE训练模块
├── utils.py               # 工具函数和辅助功能
├── requirements.txt       # 项目依赖
└── README.md             # 项目说明文档
```

## 功能特性

### 🔤 Unicode字符处理 (`unicode_demo.py`)
- 演示 `ord()` 和 `chr()` 函数的使用
- 展示Unicode字符的基本操作
- 分析不同字符的编码特性
- 演示空字符和特殊字符的行为

### 🔤 UTF-8编码处理 (`utf8_encoding.py`)
- 比较不同编码格式（UTF-8, UTF-16, UTF-32）
- 演示UTF-8字节模式
- 展示编码和解码过程
- 解释UTF-8相比其他编码的优势
- 错误处理和无效序列示例

### 🔤 BPE分词器核心 (`bpe_tokenizer.py`)
- 完整的BPE训练算法实现
- 支持自定义词汇表大小
- 特殊token处理
- 编码和解码功能
- 模型保存和加载

### 🔤 并行训练 (`parallel_bpe.py`)
- 多进程并行处理大数据集
- 自动分块和结果合并
- 内存友好的处理方式
- 支持大规模语料训练

### 🔤 工具函数 (`utils.py`)
- 日志配置和管理
- 文件操作和统计
- 性能监控
- 文本分析工具
- 进度显示

## 安装和使用

### 环境要求
- Python 3.7+
- 可选依赖：`psutil` (用于系统监控)

### 安装依赖
```bash
pip install -r requirements.txt
```

### 快速开始

#### 1. Unicode演示
```bash
python main.py demo-unicode
```

#### 2. UTF-8编码演示
```bash
python main.py demo-utf8
```

#### 3. 训练BPE模型
```bash
# 使用示例数据训练
python main.py train -i data.txt -v 300 -o vocab.txt -m merges.txt

# 指定最大合并次数
python main.py train -i data.txt -v 500 -m 200 --max-merges 150

# 添加额外特殊token
python main.py train -i data.txt -v 300 --special-tokens "<mask>,<cls>,<sep>"
```

#### 4. 测试分词器
```bash
python main.py test
```

#### 5. 分析语料库
```bash
python main.py analyze -i data.txt -o analysis.json
```

#### 6. 创建演示数据
```bash
# 创建中等大小的演示数据
python main.py create-data --size medium -o demo.txt

# 创建大数据集
python main.py create-data --size large --parallel -o big_data.txt
```

## 命令行参数

### 全局参数
- `--log-level`: 日志级别 (DEBUG, INFO, WARNING, ERROR)

### train 命令参数
- `-i, --input`: 输入训练文件 (必需)
- `-v, --vocab-size`: 词汇表大小 (默认: 300)
- `-m, --max-merges`: 最大合并次数 (默认: 100)
- `--vocab-output`: 词汇表输出文件
- `--merges-output`: 合并规则输出文件
- `--special-tokens`: 额外特殊token (逗号分隔)
- `-p, --num-processes`: 进程数 (默认: 自动检测)

## 技术细节

### BPE算法
1. **初始化**: 从256个字节token开始
2. **统计**: 计算所有相邻字节对的频率
3. **合并**: 选择频率最高的字节对进行合并
4. **迭代**: 重复直到达到目标词汇表大小

### 并行处理
- 自动将大文件分割成多个块
- 每个进程独立处理一个数据块
- 合并所有进程的统计结果
- 进行全局BPE训练

### 编码解码
- **编码**: 将文本转换为token ID序列
- **解码**: 将token ID序列转换回文本
- **特殊处理**: 未知token和特殊token的处理

## 示例输出

### Unicode演示
```
=== Unicode字符处理演示 ===
chr(0): '\x00'
chr(0)显示: 
ord('A'): 65
ord('中'): 20013
chr(65): A
chr(20013): 中
```

### UTF-8编码演示
```
=== 编码格式比较 ===

文本: 'Hello World'
UTF-8:  11 bytes - b'Hello World'
UTF-16: 24 bytes - b'\xff\xfeH\x00e\x00l\x00l\x00o\x00'
UTF-32: 48 bytes - b'\xff\xfe\x00\x00H\x00\x00\x00e\x00\x00\x00'
```

### BPE训练
```
=== 训练BPE分词器 ===
输入文件: data.txt
词汇表大小: 300
最大合并次数: 100

开始训练BPE分词器...
词汇表大小: 300
合并操作数量: 100

训练完成！词汇表大小: 300, 合并次数: 100
```

## 扩展功能

### 添加新功能
1. 创建新的Python模块
2. 在 `main.py` 中添加相应的子命令
3. 更新README文档

### 性能优化
- 使用 `numpy` 进行向量化计算
- 实现更高效的内存管理
- 添加GPU支持（使用CUDA）

### 功能增强
- 支持更多的分词算法（WordPiece, Unigram）
- 添加模型评估指标
- 支持多语言处理
- 集成深度学习框架

## 贡献

欢迎提交Issue和Pull Request来改进这个项目！

## 许可证

本项目采用MIT许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 致谢

- 感谢BPE算法的原始作者
- 感谢开源社区的贡献
- 感谢所有测试和使用者

---

**注意**: 这个项目是基于学习目的创建的，用于演示BPE分词器的工作原理。在生产环境中，建议使用成熟的分词器库如 Hugging Face Tokenizers 或 SentencePiece。
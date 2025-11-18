# 使用示例

本文档提供了BPE分词器项目的详细使用示例。

## 基础演示

### 1. Unicode字符处理演示

```bash
# 运行Unicode演示
python main.py demo-unicode

# 输出示例：
=== Unicode字符处理演示 ===
chr(0): '\x00'
chr(0)显示: 
ord('A'): 65
ord('中'): 20013
chr(65): A
chr(20013): 中
```

### 2. UTF-8编码演示

```bash
# 运行UTF-8编码演示
python main.py demo-utf8

# 输出示例：
=== 编码格式比较 ===

文本: 'Hello World'
UTF-8:  11 bytes - b'Hello World'
UTF-16: 24 bytes - b'\xff\xfeH\x00e\x00l\x00l\x00o\x00'
UTF-32: 48 bytes - b'\xff\xfe\x00\x00H\x00\x00\x00e\x00\x00\x00'
```

## 创建训练数据

### 创建不同大小的数据集

```bash
# 创建小数据集（1000行）
python main.py create-data --size small -o small_data.txt

# 创建中等数据集（10000行）
python main.py create-data --size medium -o medium_data.txt

# 创建大数据集（100000行）
python main.py create-data --size large --parallel -o large_data.txt
```

### 查看数据文件信息

```bash
# 分析语料库
python main.py analyze -i medium_data.txt -o analysis.json

# 输出示例：
=== 语料库分析 ===
文件: medium_data.txt
大小: 1.23 MB
行数: 10,000
单词数: 120,000
唯一单词: 8,500
ASCII字符: 450,000
中文字符: 150,000
其他Unicode字符: 5,000
```

## 训练BPE分词器

### 基础训练

```bash
# 基础训练
python main.py train -i medium_data.txt -v 300 -o vocab.txt -m merges.txt

# 参数说明：
# -i: 输入训练文件
# -v: 词汇表大小 (300)
# -o: 词汇表输出文件
# -m: 合并规则输出文件
```

### 高级训练选项

```bash
# 指定最大合并次数
python main.py train -i medium_data.txt -v 500 -m 200 --max-merges 150

# 添加额外特殊token
python main.py train -i medium_data.txt -v 300 \
  --special-tokens "<mask>,<cls>,<sep>" \
  -o vocab.txt -m merges.txt

# 使用多进程并行训练
python main.py train -i large_data.txt -v 1000 \
  -p 8 --max-merges 500 \
  -o vocab.txt -m merges.txt
```

### 训练输出示例

```
=== 训练BPE分词器 ===
输入文件: medium_data.txt
词汇表大小: 300
最大合并次数: 100
特殊token: ['<pad>', '<unk>', '<s>', '</s>']
输入文件大小: 1.23 MB
文件较小，使用单进程训练...

开始训练BPE分词器...
词汇表大小: 300
实际合并次数: 100

训练完成！词汇表大小: 300, 合并次数: 100
词汇表已保存到: vocab.txt
合并规则已保存到: merges.txt
```

## 测试分词器

### 交互式测试

```bash
# 启动测试模式
python main.py test

# 交互式测试示例：
=== 测试BPE分词器 ===
词汇表大小: 300
合并操作数量: 100

输入要编码的文本 (输入 'quit' 退出): Hello world!
Token IDs: [72, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100, 33]
解码结果: 'Hello world!'
Token数量: 12

输入要编码的文本 (输入 'quit' 退出): 机器学习
Token IDs: [228, 184, 173, 230, 150, 166, 229, 173, 166, 231, 137, 147]
解码结果: '机器学习'
Token数量: 12

输入要编码的文本 (输入 'quit' 退出): quit
```

### 编程方式使用

```python
from bpe_tokenizer import BPETokenizer

# 创建分词器实例
tokenizer = BPETokenizer()

# 加载训练好的模型
tokenizer.load('vocab.txt', 'merges.txt')

# 编码文本
text = "Hello world! 机器学习"
token_ids = tokenizer.encode(text)
print(f"Token IDs: {token_ids}")

# 解码回文本
decoded = tokenizer.decode(token_ids)
print(f"解码结果: {decoded}")

# 保存模型
tokenizer.save('my_vocab.txt', 'my_merges.txt')
```

## 性能测试

### 训练性能比较

```bash
# 单进程训练 (10000行)
time python main.py train -i medium_data.txt -v 300 --max-merges 100

# 多进程训练 (100000行)
time python main.py train -i large_data.txt -v 1000 -p 8 --max-merges 500
```

### 分词性能测试

```python
import time
from bpe_tokenizer import BPETokenizer

# 加载模型
tokenizer = BPETokenizer()
tokenizer.load('vocab.txt', 'merges.txt')

# 测试文本
test_text = "This is a test sentence for benchmarking the tokenizer performance."

# 性能测试
start_time = time.time()
for _ in range(1000):
    tokens = tokenizer.encode(test_text)
end_time = time.time()

avg_time = (end_time - start_time) / 1000
print(f"平均编码时间: {avg_time:.6f} 秒")
print(f"每秒处理: {1/avg_time:.0f} 个句子")
```

## 实际应用示例

### 文本预处理流水线

```python
from bpe_tokenizer import BPETokenizer
from utils import analyze_text_file

class TextPreprocessor:
    def __init__(self, vocab_file, merges_file):
        self.tokenizer = BPETokenizer()
        self.tokenizer.load(vocab_file, merges_file)
    
    def preprocess_file(self, input_file, output_file):
        """预处理文本文件"""
        with open(input_file, 'r', encoding='utf-8') as f_in:
            with open(output_file, 'w', encoding='utf-8') as f_out:
                for line in f_in:
                    line = line.strip()
                    if line:
                        tokens = self.tokenizer.encode(line)
                        f_out.write(' '.join(map(str, tokens)) + '\n')
    
    def analyze_and_preprocess(self, input_file):
        """分析并预处理文件"""
        # 分析原始文件
        original_stats = analyze_text_file(input_file)
        print(f"原始文件统计:")
        print(f"  行数: {original_stats['total_lines']:,}")
        print(f"  单词数: {original_stats['total_words']:,}")
        print(f"  唯一单词: {original_stats['unique_words']:,}")
        
        # 预处理
        preprocessed_file = input_file + '.tokens'
        self.preprocess_file(input_file, preprocessed_file)
        
        # 分析预处理后的文件
        token_stats = analyze_text_file(preprocessed_file)
        print(f"\n预处理后统计:")
        print(f"  平均token数: {token_stats['total_words'] / token_stats['total_lines']:.1f}")
        print(f"  唯一token数: {token_stats['unique_words']:,}")

# 使用示例
preprocessor = TextPreprocessor('vocab.txt', 'merges.txt')
preprocessor.analyze_and_preprocess('medium_data.txt')
```

### 多语言支持

```python
from bpe_tokenizer import BPETokenizer

class MultilingualTokenizer:
    def __init__(self):
        self.tokenizers = {}
    
    def train_language(self, lang_code, training_file, vocab_size=300):
        """为特定语言训练分词器"""
        tokenizer = BPETokenizer()
        tokenizer.train(
            input_path=training_file,
            vocab_size=vocab_size,
            special_tokens=['<pad>', '<unk>', f'<{lang_code}>']
        )
        self.tokenizers[lang_code] = tokenizer
        return tokenizer
    
    def encode_multilingual(self, text, lang_code=None):
        """编码多语言文本"""
        if lang_code and lang_code in self.tokenizers:
            return self.tokenizers[lang_code].encode(text)
        else:
            # 使用默认分词器或自动检测
            return self.tokenizers['en'].encode(text)
    
    def save_all(self, output_dir):
        """保存所有语言的分词器"""
        os.makedirs(output_dir, exist_ok=True)
        for lang, tokenizer in self.tokenizers.items():
            vocab_file = os.path.join(output_dir, f'vocab_{lang}.txt')
            merges_file = os.path.join(output_dir, f'merges_{lang}.txt')
            tokenizer.save(vocab_file, merges_file)

# 使用示例
multi_tokenizer = MultilingualTokenizer()

# 训练英语分词器
multi_tokenizer.train_language('en', 'english_corpus.txt', 500)

# 训练中文分词器  
multi_tokenizer.train_language('zh', 'chinese_corpus.txt', 500)

# 编码多语言文本
english_tokens = multi_tokenizer.encode_multilingual("Hello world!", 'en')
chinese_tokens = multi_tokenizer.encode_multilingual("你好世界！", 'zh')

print(f"英语token数: {len(english_tokens)}")
print(f"中文token数: {len(chinese_tokens)}")
```

## 故障排除

### 常见问题

1. **Unicode解码错误**
   ```bash
   # 确保文件使用UTF-8编码
   python main.py analyze -i file.txt
   ```

2. **内存不足**
   ```bash
   # 使用并行处理减少内存占用
   python main.py train -i large_file.txt -p 8 --max-merges 200
   ```

3. **训练时间过长**
   ```bash
   # 限制合并次数
   python main.py train -i data.txt -v 300 --max-merges 50
   ```

### 性能优化建议

1. **大数据集处理**
   - 使用并行训练 (`-p` 参数)
   - 适当增加进程数
   - 限制合并次数

2. **内存优化**
   - 分批处理大文件
   - 使用生成器处理数据
   - 及时清理临时文件

3. **训练加速**
   - 预过滤低频词汇
   - 使用更高效的数据结构
   - 考虑使用Cython优化关键代码

## 最佳实践

1. **数据预处理**
   - 清理文本中的特殊字符
   - 统一编码格式 (UTF-8)
   - 去除多余空白

2. **参数选择**
   - 词汇表大小: 根据语料库大小调整 (300-50000)
   - 合并次数: 通常为目标词汇表大小的1/3
   - 特殊token: 根据任务需求添加

3. **模型评估**
   - 检查词汇表覆盖率
   - 分析token分布
   - 测试编码解码一致性

---

这些示例涵盖了从基础使用到高级应用的各种场景。你可以根据具体需求调整参数和代码。如有问题，请参考主程序的帮助信息或查看源代码。
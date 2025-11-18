#!/usr/bin/env bash
# CS336-assignment1 一键整理脚本
# 用法：bash reorganize.sh
set -euo pipefail

########## 0. 备份当前状态（若已 git init） ##########
if [ -d .git ]; then
    echo ">>> 检测到 Git 仓库，先 stash 当前变更以便回滚"
    git stash push -m "before-reorganize"
fi

########## 1. 修正拼写 ##########
if [ -f reguirements.txt ]; then
    echo ">>> 重命名 reguirements.txt -> requirements.txt"
    mv reguirements.txt requirements.txt
fi

########## 2. 创建目录 ##########
echo ">>> 创建分层文件夹"
mkdir -p src/{tokenizers,encoders,optimizers} tests notebooks scripts docs

########## 3. 移动文件 ##########
echo ">>> 移动文件到对应目录"

# tokenizers
mv bpe_tokenizer.py  src/tokenizers/
mv parallel_bpe.py   src/tokenizers/

# encoders
mv utf8_encoding.py  src/encoders/
mv unicode_demo*.py  src/encoders/

# optimizers
mv dl_optimizers.py  src/optimizers/

# utils
mv utils.py          src/

# tests
mv test_*.py         tests/

# notebooks
mv *.ipynb           notebooks/

# scripts
mv main.py           scripts/

# docs
mv *.md              docs/
mv src/encoders/unicode_demo*.py docs/ 2>/dev/null || true   # 如果希望 demo 放 docs 可取消注释

########## 4. 把 src 标记为 Python 包 ##########
touch src/__init__.py
touch src/tokenizers/__init__.py
touch src/encoders/__init__.py
touch src/optimizers/__init__.py

########## 5. 统一 import 路径 ##########
echo ">>> 修正 Python 文件中的 import 路径"

# 需要替换的原始导入（根据你实际代码调整）
declare -A OLD2NEW
OLD2NEW["from bpe_tokenizer"]="from src.tokenizers.bpe_tokenizer"
OLD2NEW["from parallel_bpe"]="from src.tokenizers.parallel_bpe"
OLD2NEW["from utf8_encoding"]="from src.encoders.utf8_encoding"
OLD2NEW["from unicode_demo"]="from src.encoders.unicode_demo"
OLD2NEW["from dl_optimizers"]="from src.optimizers.dl_optimizers"
OLD2NEW["from utils"]="from src.utils"

# 递归替换
while IFS= read -r -d '' pyfile; do
    for old in "${!OLD2NEW[@]}"; do
        new=${OLD2NEW[$old]}
        # macOS 兼容写法
        sed -i.bak "s/^${old} /${new} /g" "$pyfile"
        rm -f "${pyfile}.bak"
    fi
done < <(find . -name "*.py" -type f)

########## 6. 提示 ##########
echo ">>> 整理完成！结构如下："
tree -L 3 -F 2>/dev/null || find . -type f | sort

echo ""
echo ">>> 验证测试能否通过："
echo "    python -m pytest tests/          # 跑测试"
echo "    python -m scripts.main           # 跑主脚本"
echo ""
echo ">>> 若需回滚：git stash pop"

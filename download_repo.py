import requests
import zipfile
import io
import os
import shutil

# GitHub仓库ZIP下载链接
url = 'https://github.com/Xuan-0929/cs336-assignment1/archive/refs/heads/main.zip'

# 下载ZIP文件
print('正在下载仓库内容...')
r = requests.get(url)

# 解压ZIP文件
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall('.')

# 移动文件到当前目录
source_dir = 'cs336-assignment1-main'
print(f'正在提取文件到当前目录...')

if os.path.exists(source_dir):
    for item in os.listdir(source_dir):
        source = os.path.join(source_dir, item)
        destination = item
        # 如果目标文件已存在，先删除
        if os.path.exists(destination):
            if os.path.isdir(destination):
                shutil.rmtree(destination)
            else:
                os.remove(destination)
        shutil.move(source, destination)
    
    # 删除临时目录
    os.rmdir(source_dir)
    print('仓库内容已成功下载并提取到当前目录！')
else:
    print(f'错误：解压后的目录 {source_dir} 不存在')

"""
下载项目所需的模型到 models/ 目录
模型来源: Hugging Face
- BAAI/bge-m3 (向量检索)
- BAAI/bge-reranker-large (重排序)
- google-bert/bert-base-chinese (查询分类)

使用方法:
    pip install huggingface_hub
    python download_models.py

国内加速 (使用 hf-mirror):
    HF_ENDPOINT=https://hf-mirror.com python download_models.py
    Windows: set HF_ENDPOINT=https://hf-mirror.com && python download_models.py
"""

import os
from huggingface_hub import snapshot_download

MODELS = [
    ("BAAI/bge-m3", "models/bge-m3"),
    ("BAAI/bge-reranker-large", "models/bge-reranker-large"),
    ("google-bert/bert-base-chinese", "models/bert-base-chinese"),
]

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    for repo_id, local_dir in MODELS:
        target = os.path.join(ROOT_DIR, local_dir)
        if os.path.exists(target) and os.listdir(target):
            print(f"[跳过] {repo_id} 已存在: {target}")
            continue
        print(f"[下载] {repo_id} -> {target}")
        snapshot_download(
            repo_id=repo_id,
            local_dir=target,
            local_dir_use_symlinks=False,
        )
        print(f"[完成] {repo_id}")

    print("\n所有模型下载完成。")


if __name__ == "__main__":
    main()

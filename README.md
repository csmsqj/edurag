# EduRAG - 教育智慧问答系统

基于 BM25 + 向量检索的双路径教育知识问答系统，支持中文文档解析、混合检索和语义重排序。

## 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                      用户查询                            │
└──────────────┬──────────────────────┬───────────────────┘
               │                      │
       ┌───────▼───────┐      ┌───────▼───────┐
       │  BM25 检索路径  │      │  向量检索路径  │
       │  (mysql_all/)  │      │    (rag/)     │
       └───────┬───────┘      └───────┬───────┘
               │                      │
        CSV → MySQL              文档加载 + 分块
               │                      │
        Redis 缓存             BGE-M3 向量化
               │                      │
        jieba 分词 + BM25       Milvus 混合检索
               │                      │
        softmax 阈值过滤        CrossEncoder 重排序
               │                      │
       ┌───────▼──────────────────────▼───────┐
       │              LLM 生成回答              │
       └───────────────────────────────────────┘
```

## 快速开始

### 1. 环境要求

- Python 3.9+
- MySQL 5.7+（端口 3306）
- Redis 6.0+（端口 6379）
- Milvus 2.x（端口 19530，仅向量路径需要）

### 2. 安装依赖

```bash
pip install -r requirements.txt -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

### 3. 下载模型

向量检索路径需要以下本地模型（约 12 GB）：

| 模型 | 用途 | 来源 |
|------|------|------|
| BAAI/bge-m3 | 向量编码（dense + sparse） | [Hugging Face](https://huggingface.co/BAAI/bge-m3) |
| BAAI/bge-reranker-large | 语义重排序 | [Hugging Face](https://huggingface.co/BAAI/bge-reranker-large) |
| google-bert/bert-base-chinese | 查询意图分类 | [Hugging Face](https://huggingface.co/google-bert/bert-base-chinese) |

一键下载：

```bash
pip install huggingface_hub

# 国内用户使用镜像加速
set HF_ENDPOINT=https://hf-mirror.com
python download_models.py
```

模型会下载到项目根目录的 `models/` 文件夹。

### 4. 配置环境变量

创建 `.env` 文件：

```env
REDIS_PASSWORD=your_redis_password
MYSQL_PASSWORD=your_mysql_password
DEEPSEEK_API_KEY=your_api_key
```

### 5. 运行

**BM25 路径：**

```bash
# 初始化 MySQL 表并导入 CSV 数据
python mysql_all/mysql/mysql_client.py

# 启动问答交互
python mysql_all/main.py
```

**向量检索路径：**

```bash
# 加载文档并分块入库
python rag/core/document_loader_splitter.py
```

## 项目结构

```
bs25_mysql_redis/
├── base/                        # 基础配置
│   ├── config.py                # 配置读取 (config.ini + 环境变量)
│   └── log.py                   # 日志配置
├── mysql_all/                   # BM25 检索路径
│   ├── mysql/mysql_client.py    # MySQL 数据层
│   ├── cache/redis_client.py    # Redis 缓存层
│   ├── retrieval/bm25_search.py # BM25 检索引擎
│   ├── utils/test_preProcess.py # jieba 分词预处理
│   └── main.py                  # 入口
├── rag/core/                    # 向量检索路径
│   ├── document_loader_splitter.py  # 文档加载 + 父子分块
│   ├── vector_store.py          # Milvus 存储 + BGE-M3 编码 + 重排序
│   └── query_classifier.py      # 查询意图分类
├── edu_document_loaders/        # 自定义文档加载器 (PDF/PPT/DOC/图片)
├── edu_text_splitter/           # 中文递归分块器
├── models/                      # 本地模型 (git ignored)
├── data/                        # 数据文件
├── config.ini                   # 配置文件
├── download_models.py           # 模型下载脚本
└── requirements.txt             # 依赖清单
```

## 检索参数

在 `config.ini` 中调整：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| parent_chunk_size | 512 | 父块大小（字符），作为 LLM 上下文 |
| child_chunk_size | 128 | 子块大小（字符），用于精确检索 |
| chunk_overlap | 50 | 块间重叠字符数 |
| retrieval_k | 3 | 最终返回结果数 |
| candidate_m | 2 | 初步召回候选数（重排序前） |

## 技术栈

- **检索**: BM25L / Milvus hybrid search (dense + sparse)
- **向量模型**: BGE-M3
- **重排序**: BGE-Reranker-Large (CrossEncoder)
- **文档解析**: PyMuPDF + RapidOCR + python-pptx + python-docx
- **中文分词**: jieba
- **框架**: LangChain
- **缓存**: Redis
- **存储**: MySQL / Milvus

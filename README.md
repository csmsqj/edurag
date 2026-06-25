# EduRAG - 教育智慧问答系统

基于 BM25 + RAG 双路径的教育知识问答系统。支持中文文档解析、BERT 意图分类、多策略检索增强、混合向量检索和语义重排序。

## 一、项目做了什么

本项目实现了一个面向教育场景的智能问答系统，核心能力：

1. **BM25 关键词检索路径**：将 CSV 格式的 Q&A 知识库导入 MySQL，通过 jieba 分词 + BM25L 算法进行关键词匹配检索，Redis 缓存加速。
2. **RAG 向量检索路径**：支持多格式文档（PDF/PPT/DOC/MD/图片）加载，父子分块策略，BGE-M3 模型生成稠密+稀疏双向量，Milvus 混合检索，CrossEncoder 重排序。
3. **智能查询分类**：使用微调的 BERT 模型自动判断用户问题属于"通用知识"还是"专业咨询"，通用知识直接由 LLM 回答，专业问题走 RAG 检索流程。
4. **多策略检索增强**：根据问题特征自动选择最优检索策略 —— 直接检索、HyDE（假设文档嵌入）、子查询分解、回溯问题简化。
5. **LLM 答案生成**：基于检索到的上下文，调用大模型生成最终回答。

## 二、系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                        用户查询                              │
└────────────────────────────┬────────────────────────────────┘
                             │
                   ┌─────────▼─────────┐
                   │  BERT 意图分类器    │
                   │ (通用知识/专业咨询)  │
                   └────┬─────────┬────┘
                        │         │
            ┌───────────▼──┐  ┌──▼────────────────┐
            │  通用知识      │  │  专业咨询          │
            │  直接调 LLM   │  │  策略选择器        │
            └───────┬──────┘  └──┬────────────────┘
                    │            │
                    │   ┌────────▼────────────────────┐
                    │   │ 检索策略（四选一）            │
                    │   │ · 直接检索                   │
                    │   │ · HyDE 假设文档嵌入          │
                    │   │ · 子查询分解                 │
                    │   │ · 回溯问题简化               │
                    │   └────────┬────────────────────┘
                    │            │
                    │   ┌────────▼────────┐
                    │   │ Milvus 混合检索  │
                    │   │ (dense + sparse) │
                    │   └────────┬────────┘
                    │            │
                    │   ┌────────▼────────────┐
                    │   │ CrossEncoder 重排序  │
                    │   └────────┬────────────┘
                    │            │
            ┌───────▼────────────▼───────┐
            │       LLM 生成最终回答       │
            └────────────────────────────┘
```

**BM25 检索路径（独立使用）：**

```
CSV 数据 → MySQL 存储 → Redis 缓存分词结果 → jieba 分词 → BM25L 评分 → softmax 阈值过滤 → 返回答案
```

## 三、如何使用

### 1. 环境要求

- Python 3.9+
- MySQL 5.7+（端口 3306）
- Redis 6.0+（端口 6379）
- Milvus 2.x（端口 19530，仅向量检索路径需要）

### 2. 安装依赖

```bash
pip install -r requirements.txt -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

### 3. 下载本地模型

向量检索路径需要以下本地模型（约 12 GB）：

| 模型 | 用途 | 来源 |
|------|------|------|
| BAAI/bge-m3 | 向量编码（dense + sparse） | [HuggingFace](https://huggingface.co/BAAI/bge-m3) |
| BAAI/bge-reranker-large | 语义重排序 | [HuggingFace](https://huggingface.co/BAAI/bge-reranker-large) |
| google-bert/bert-base-chinese | 查询意图分类 | [HuggingFace](https://huggingface.co/google-bert/bert-base-chinese) |

一键下载（国内推荐使用镜像）：

```bash
set HF_ENDPOINT=https://hf-mirror.com
python download_models.py
```

模型会下载到项目根目录的 `models/` 文件夹。

### 4. 配置

**环境变量**：创建 `.env` 文件：

```env
REDIS_PASSWORD=your_redis_password
MYSQL_PASSWORD=your_mysql_password
DEEPSEEK_API_KEY=your_api_key
```

**系统配置**：`config.ini` 中可调整检索参数、数据库连接、LLM 模型等。

### 5. 运行 BM25 路径

```bash
# 第一步：初始化 MySQL 表并导入 CSV 数据
python mysql_all/mysql/mysql_client.py

# 第二步：启动 BM25 问答交互
python mysql_all/main.py
```

### 6. 运行 RAG 向量检索路径

```bash
# 第一步：加载文档到 Milvus 向量库（数据处理模式）
python rag/core/rag_main.py --directory_path ./data

# 第二步：启动 RAG 问答交互（查询模式）
python rag/core/rag_main.py --query_mode
```

命令行参数说明：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--query_mode` | 加此参数进入问答模式，不加则进入数据处理模式 | False（数据处理） |
| `--directory_path` | 文档目录路径 | `./data` |

## 四、项目结构

```
bs25_mysql_redis/
├── base/                            # 基础设施
│   ├── config.py                    # 配置读取（config.ini + 环境变量）
│   ├── log.py                       # 日志（控制台 INFO / 文件 WARN+）
│   └── output_parser.py             # LLM 输出后处理
├── mysql_all/                       # BM25 检索路径
│   ├── mysql/mysql_client.py        # MySQL 数据层（表 jpkb）
│   ├── cache/redis_client.py        # Redis 缓存（分词结果 + 查询答案）
│   ├── retrieval/bm25_search.py     # BM25L 检索引擎
│   ├── utils/test_preProcess.py     # jieba 分词预处理
│   └── main.py                      # BM25 路径入口
├── rag/                             # RAG 向量检索路径
│   ├── core/
│   │   ├── rag_main.py              # RAG 路径入口（命令行）
│   │   ├── rag_system.py            # RAG 核心（分类→策略→检索→生成）
│   │   ├── query_classifier.py      # BERT 查询意图分类器
│   │   ├── strategy_selector.py     # 检索策略选择器
│   │   ├── vector_store.py          # Milvus 存储 + BGE-M3 编码 + 重排序
│   │   ├── document_loader_splitter.py  # 文档加载 + 父子分块
│   │   └── prompts.py               # Prompt 模板（RAG/HyDE/子查询/回溯）
│   └── classify_data/               # 意图分类训练数据
├── edu_document_loaders/            # 自定义文档加载器（PDF/PPT/DOC/图片）
├── edu_text_splitter/               # 中文递归分块器（中文标点分割）
├── models/                          # 本地模型目录（不提交 Git）
├── data/                            # 知识库文档
├── config.ini                       # 系统配置文件
├── download_models.py               # 模型下载脚本
└── requirements.txt                 # Python 依赖
```

## 五、核心检索参数

在 `config.ini` 的 `[retrieval]` 段中配置：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| parent_chunk_size | 512 | 父块大小（字符），作为 LLM 上下文 |
| child_chunk_size | 128 | 子块大小（字符），用于精确检索匹配 |
| chunk_overlap | 50 | 块间重叠字符数，防止切断语义 |
| retrieval_k | 3 | 最终返回的文档块数量 |
| candidate_m | 2 | 初步召回候选数（重排序前） |

## 六、技术栈

| 类别 | 技术 |
|------|------|
| 关键词检索 | BM25L + jieba 分词 |
| 向量检索 | Milvus hybrid search（dense + sparse） |
| 向量模型 | BGE-M3（本地推理） |
| 重排序 | BGE-Reranker-Large（CrossEncoder） |
| 意图分类 | BERT-base-chinese（微调） |
| 文档解析 | PyMuPDF + RapidOCR + python-pptx + python-docx |
| LLM | 兼容 OpenAI API 的大模型（可配置） |
| 框架 | LangChain |
| 缓存 | Redis |
| 存储 | MySQL + Milvus |

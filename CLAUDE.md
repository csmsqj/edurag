# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EduRAG - an educational Q&A system with two retrieval paths:
1. **BM25 path** (`mysql_all/`): CSV → MySQL → Redis cache → BM25 ranking → answer
2. **Vector path** (`rag/`): Documents → parent/child chunking → BGE-M3 embeddings → Milvus hybrid search → CrossEncoder reranking

## Running the Project

```bash
# Install dependencies
pip install -r requirements.txt -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# Initialize MySQL table and import CSV data
python mysql_all/mysql/mysql_client.py

# Run BM25 Q&A interactive loop
python mysql_all/main.py

# Run document loading + chunking pipeline
python rag/core/document_loader_splitter.py
```

No test framework or linter is configured. Individual modules have `if __name__ == '__main__'` blocks for manual testing.

## Architecture

### Configuration (`base/`)
- `config.py`: Reads `config.ini` (fixed params) + environment variables for secrets (REDIS_PASSWORD, MYSQL_PASSWORD, DEEPSEEK_API_KEY). All modules instantiate `Config()` to access settings.
- `log.py`: `get_logger()` returns a shared logger. Console output at INFO, file output (`app/app.log`) at WARN+.

### BM25 Retrieval Pipeline (`mysql_all/`)
- `mysql/mysql_client.py`: PyMySQL wrapper. Table `jpkb` with columns (subject_name, question, answer). Data imported from CSV.
- `cache/redis_client.py`: Redis wrapper using JSON serialization. Caches tokenized questions (key: `qa_tokenized_questions`) and query answers (key: `answer:{query}`).
- `retrieval/bm25_search.py`: On init, loads questions from Redis (fallback: MySQL → tokenize with jieba → cache to Redis). Search: tokenize query → BM25L scoring → softmax → threshold check → return answer.
- `utils/test_preProcess.py`: `preprocess_text()` - jieba segmentation + lowercase.

### RAG Document Pipeline (`rag/core/`)
- `document_loader_splitter.py`: Loads files from a directory using format-specific loaders, then applies two-level chunking (parent chunks → child chunks). Markdown uses `MarkdownTextSplitter`; other formats use `ChineseRecursiveTextSplitter`.
- `vector_store.py`: Milvus client with BGE-M3 for hybrid (dense + sparse) embeddings. Uses `models/bge-m3` and `models/bge-reranker-large` local model directories.

### Custom Document Loaders (`edu_document_loaders/`)
Loaders for PDF (with OCR fallback via RapidOCR), PPT, DOC/DOCX, and images. All implement LangChain's `BaseLoader` interface with `lazy_load()`.

### Chinese Text Splitter (`edu_text_splitter/`)
`ChineseRecursiveTextSplitter`: extends LangChain's `RecursiveCharacterTextSplitter` with Chinese punctuation separators (。？！；，).

## Key Conventions

- Import paths use `sys.path.insert(0, ...)` at module top since there's no installed package. Modules are run from their own directory or from project root.
- Config values: `config.ini` for non-sensitive params, environment variables (with hardcoded fallback defaults) for passwords/API keys.
- MySQL database: `edurag`, table: `jpkb`.
- Redis DB: 15.
- Retrieval params (from config.ini): parent_chunk_size=512, child_chunk_size=128, chunk_overlap=50, retrieval_k=3, candidate_m=2.
- LLM config points to DeepSeek API (model: deepseek-v4-flash) but is not yet wired into the retrieval pipeline.

## External Dependencies

- **Services required**: MySQL (port 3306), Redis (port 6379), Milvus (port 19530, only for vector path)
- **Local models** (for vector path): `models/bge-m3`, `models/bge-reranker-large`
- **Data**: `data/JP学科知识问答.csv` is the source for the BM25 Q&A knowledge base

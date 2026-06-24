# ===== 数据处理 =====
import pandas as pd
# pandas：数据分析库，用于把评估结果转成 DataFrame 保存为 CSV 文件
import json
"""
#"D:\eduEag笔记\EduRAG评估系统笔记7.md"，帮我优化这份笔记，
#要求1：笔记要是有不符合我原本项目的变量名你可以修改
#2主要要求：优化笔记细节：比如这个地方**各依赖包安装命令汇总**：
pip install ragas              # RAGAS 评估框架
pip install datasets           # HuggingFace datasets（提供 Dataset 类）
pip install pandas             # 数据处理
pip install langchain-openai   # LangChain 的 OpenAI 兼容封装
pip install openai             # OpenAI SDK（langchain-openai 的底层依赖）应该说明清楚langchain-openai和openai对应哪个地方要导入？
也就是更加细节，详细
3rag_evaluate_data.json"这个我文件我没有，评估系统文件需要你根据我的项目生成一个大概的了
"""

#rag_evaluate_data.json
#context_relevancy（上下文相关性）
#context_recall（上下文召回率）
#faithfulness（忠实度）
#answer_relevancy（答案相关性）
from ragas.metrics import (
    faithfulness,       # 忠实度：答案是否基于上下文
    answer_relevancy,   # 答案相关性：答案是否与问题匹配
    context_relevancy,  # 上下文相关性：上下文是否仅包含相关信息
    context_recall      # 上下文召回率：上下文是否包含所有必要信息
)

# ===== HuggingFace Datasets =====
from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# 1. 加载生成的数据集
# 使用 with 语句打开 JSON 文件，确保文件正确关闭，指定编码为 utf-8
with open("rag_evaluate_data.json", "r", encoding="utf-8") as f:
    # 将 JSON 文件内容加载到 data 变量中
    # data 是一个列表，每个元素是一个字典，包含 question/context/answer/ground_truth 四个字段
    data = json.load(f)

# data 的结构示例：
# [
#     {"question": "人工智能就业课的课程版本是什么？",
#      "context": ["人工智能学科全新升级—人工智能开发V6.0课程。"],
#      "answer": "人工智能就业课的课程版本是V6.0。",
#      "ground_truth": "V6.0"},
#     {"question": "课程优势有哪些？", ...},
#     ...
# ]


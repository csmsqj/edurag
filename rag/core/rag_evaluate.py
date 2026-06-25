import os
import sys
rootfile=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0,rootfile )
from base.config import Config
from base.log import get_logger
from langchain_core.prompts import ChatPromptTemplate
config=Config()
logger=get_logger()
# ===== 数据处理 =====
import pandas as pd
# pandas：数据分析库，用于把评估结果转成 DataFrame 保存为 CSV 文件
import json

#rag_evaluate_data.json
#context_precision（上下文精确度）
#context_recall（上下文召回率）
#faithfulness（忠实度）
#answer_relevancy（答案相关性）
from ragas.metrics import (
    faithfulness,       # 忠实度：答案是否基于上下文
    answer_relevancy,   # 答案相关性：答案是否与问题匹配
    context_precision,  # 上下文相关性：上下文是否仅包含相关信息
    context_recall      # 上下文召回率：上下文是否包含所有必要信息
)
from ragas import evaluate


# ===== HuggingFace Datasets =====
from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

def rag_evaluate(llm, embeddings, file="rag_evaluate_data.json"):
    # 1. 加载生成的数据集
    # 使用 with 语句打开 JSON 文件，确保文件正确关闭，指定编码为 utf-8
    with open(file, "r", encoding="utf-8") as f:
        # 将 JSON 文件内容加载到 data 变量中
        # data 是一个列表，每个元素是一个字典，包含 question/context/answer/ground_truth 四个字段
        data = json.load(f)
    # 2. 转换为 RAGAS 格式
    # RAGAS 的 evaluate() 函数要求输入是 HuggingFace 的 Dataset 对象
    # Dataset.from_dict() 要求传入字典，key 是字段名，value 是该字段所有值的列表，返回值是 Dataset 对象
    eval_data = {
        # 提取每个数据条目的 question 字段，组成问题列表
        "question": [item["question"] for item in data],
        # 提取每个数据条目的 answer 字段，组成答案列表
        "answer": [item["answer"] for item in data],
        # 提取每个数据条目的 context 字段，组成上下文列表（每个 context 本身是列表）
        "contexts": [item["context"] for item in data],
        # 提取每个数据条目的 ground_truth 字段，组成真实答案列表
        "ground_truth": [item["ground_truth"] for item in data]
    }

    # 使用 Dataset.from_dict 将字典转换为 RAGAS 所需的 Dataset 对象
    dataset = Dataset.from_dict(eval_data)
    # 3调用 RAGAS 的核心函数 evaluate()
    result = evaluate(
        # 传入转换好的 Dataset 对象（第 2 步的输出）
        dataset=dataset,
        # 指定使用的评估指标列表
        metrics=[
            faithfulness,  # 忠实度：答案是否基于上下文
            answer_relevancy,  # 答案相关性：答案与问题的匹配度
            context_precision,  # 上下文相关性：上下文是否仅包含相关信息
            context_recall  # 上下文召回率：上下文是否包含所有必要信息
        ],
        # 传入配置好的 LLM 模型
        llm=llm,
        # 传入配置好的嵌入模型
        embeddings=embeddings
    )
    return result


if __name__=="__main__":
    llm=ChatOpenAI(model=config.MODEL,
                   base_url=config.URL,
                   api_key=config.DEEPSEEK_APIKEY,
                     temperature=0.2
                   )

    # ===== 配置嵌入模型
    embeddings = OpenAIEmbeddings(
        model=config.EBMEDDING_MODEL,
        api_key=config.DEEPSEEK_APIKEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        check_embedding_ctx_length=False
    )
    result=rag_evaluate(llm, embeddings, file="rag_evaluate_data.json")
    print("RAGAS 评估结果：{}".format(result))









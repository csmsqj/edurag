# ===== 标准库 =====
import json       # JSON 解析，用于读取训练数据文件（JSONL 格式）
import os          # 文件/目录操作，用于检查模型路径是否存在

# ===== PyTorch =====
import torch       # 深度学习框架，提供张量运算、设备管理、梯度控制

# ===== 项目内部模块 =====
import os
import sys
rootfile=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, rootfile)
from base.log import get_logger

# ===== 科学计算 =====
import numpy as np  # 数组运算库，compute_metrics 中用 np.argmax 取最大值索引

# ===== HuggingFace Transformers =====
from transformers import BertTokenizer                 # BERT 分词器：文字 → 数字
from transformers import BertForSequenceClassification  # BERT 分类模型：在 BERT 上加了分类层
from transformers import Trainer, TrainingArguments     # 训练工具：简化训练流程

# ===== Scikit-learn =====
from sklearn.model_selection import train_test_split     # 数据划分：把数据拆成训练集+验证集
from sklearn.metrics import classification_report        # 分类报告：输出精确率/召回率/F1
from sklearn.metrics import confusion_matrix             # 混淆矩阵：预测对错的交叉统计表


class QueryClassifier:
    def __init__(self, model_path='bert-model'):
        model_path = os.path.join(rootfile, model_path)
        self.model_path = model_path
        self.logger = get_logger(__name__)
        # 加载分词器和语义分类模型
        loader_path=os.path.join(model_path, 'bert_base_chinese')

        self.tokenizer = BertTokenizer.from_pretrained(loader_path)

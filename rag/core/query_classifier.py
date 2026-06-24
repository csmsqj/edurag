# ===== 标准库 =====
import json       # JSON 解析，用于读取训练数据文件（JSONL 格式）

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
        file=os.path.join(rootfile,"models")
        #MODEL_PATH = "bert-model"  # 模型保存路径
        model_path = os.path.join(file, model_path)
        self.model_path = model_path
        self.logger = get_logger(__name__)
        # 加载分词器和语义分类模型,loader_path:bert_base_chinese
        self.loader_path=os.path.join(file, 'bert-base-chinese')

        self.tokenizer = BertTokenizer.from_pretrained(self.loader_path)
        #模型要判断是否预训练，先设置为None，后续加载模型时再赋值
        self.model =None

        # torch.cuda.is_available() 检查是否有 NVIDIA GPU
        # 有 GPU → "cuda"，没有 → "cpu"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"使用设备: {self.device}")
        self.label_map = {"通用知识": 0, "专业咨询": 1}
        # 加载模型
        self._load_model()

    #加载模型的函数，包含两种情况：目录存在（加载）和目录不存在（初始化）
    def _load_model(self):
        """加载或初始化模型"""
        # 检查微调后的模型目录是否存在
        if os.path.exists(self.model_path):
            # 【情况 A】：目录存在 → 说明之前已经训练并保存过模型 → 直接加载
            self.model = BertForSequenceClassification.from_pretrained(self.model_path)
            # from_pretrained 会自动读取目录下的 config.json 和 model.safetensors
            self.model.to(self.device)  # 把模型搬到 CPU/GPU 上
            self.logger.info(f"加载模型: {self.model_path}")
        else:
            # 【情况 B】：目录不存在 → 说明还没训练过 → 用预训练模型初始化
            self.model = BertForSequenceClassification.from_pretrained(
                self.loader_path,  # 从原始 BERT 加载
                num_labels=2  # ⬅️ 关键参数：2 个分类（通用/专业）
                # num_labels=2 会在 BERT 的最后一层加上一个 768→2 的全连接层
                # BERT 输出 768 维向量 → 全连接层 → 输出 2 个分数（通用的分、专业的分）
            )
            self.model.to(self.device)
            self.logger.info("初始化新 BERT 模型")
            self.train_model()


    #保存微调后的模型
    def save_model(self):
        """保存微调后的模型到指定目录"""
        # save_pretrained 会在目录下生成：
        # config.json, model.safetensors, tokenizer_config.json, vocab.txt 等
        self.model.save_pretrained(self.model_path)
        self.logger.info(f"模型保存至: {self.model_path}")


    #preprocess_data 数据预处理函数，负责把原始文本和标签转换成模型输入格式(使用分词器模型)
    def preprocess_data(self, texts, labels):
        """把原始文本和标签转成模型能读懂的格式"""
        # self.tokenizer(...) 就是调用分词器，把文本列表批量编码
        encodings = self.tokenizer(
            texts,  # 文本列表，如 ["学费多少", "什么是AI", ...]
            truncation=True,  # 超过 max_length 就截断
            padding=True,  # 不够 max_length 就用 0 填充
            max_length=128,  # 每条文本最多 128 个 token
            return_tensors="pt"  # 返回 PyTorch 张量格式（而不是普通列表）
        )
         #encodings 包含：
        # input_ids: shape [N, 128]      ← N 条文本，每条 128 个 token ID
        # attention_mask: shape [N, 128] ← 1=真实token, 0=padding
        # token_type_ids: shape [N, 128] ← 全 0（单句分类）

        # 把文字标签转成数字标签
        # self.label_map = {"通用知识": 0, "专业咨询": 1}
        # 例如：labels = ["专业咨询", "通用知识", "专业咨询"]
        #       → [self.label_map["专业咨询"], self.label_map["通用知识"], ...]
        #       → [1, 0, 1]
        return encodings, [self.label_map[label] for label in labels]


    #创建pytorch数据集类，负责把预处理后的数据封装成 PyTorch Dataset 对象，方便训练时按批次加载
    def create_dataset(self, encodings, labels):
        """把编码后的数据包装成 PyTorch Dataset 对象"""
        # 在方法内部定义了一个 Dataset 内部类
        class Dataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __getitem__(self, idx):
                # 取第 idx 条数据
                item = {key: val[idx] for key, val in self.encodings.items()}
                item["labels"] = torch.tensor(self.labels[idx])
                return item

            def __len__(self):
                return len(self.labels)

        return Dataset(encodings, labels)

    #预训练模型并且评分函数
    def train_model(self):
        #1 读取训练数据
        absfile=os.path.abspath(__file__)
        dirfile=os.path.dirname(os.path.dirname(absfile))

        file=os.path.join(dirfile,"classify_data\\model_generic_500.json")
        print(file)
        if not os.path.exists(file):
            self.logger.error(f"训练数据文件不存在: {file}")
            raise FileNotFoundError(f"训练数据文件不存在: {file}")
        with open(file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        data=[]
        for line in lines:
            l = json.loads(line)
            data.append(l)
        texts = [item["query"] for item in data]
        labels = [item["label"] for item in data]
        #数据划分：80% 训练 + 20% 验证 =====
        #返回：4 个列表 → 4000 条训练文本、1000 条验证文本、4000 条训练标签、1000 条验证标签
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )
        # =====预处理：文本 → 分词编码 + 标签 → 数字 =====
        # 作用：把人类能读的文本和标签，转成模型能处理的数字格式(包括训练和验证数据)
        train_encodings, train_labels = self.preprocess_data(train_texts, train_labels)
        val_encodings, val_labels = self.preprocess_data(val_texts, val_labels)

        # =====创建 PyTorch 数据集对象 =====
        # 作用：把分词编码和数字标签包装成 Trainer 认识的 Dataset 格式，
        #       Trainer 训练时会通过 dataset[idx] 逐条取数据
        train_dataset = self.create_dataset(train_encodings, train_labels)
        val_dataset = self.create_dataset(val_encodings, val_labels)

        # ===== ⑤ 配置训练参数 =====
        # 使用 HuggingFace 的 TrainingArguments
        # 传入：各种超参数配置
        # 返回：一个 TrainingArguments 对象，传给下面的 Trainer
        training_args = TrainingArguments(
            output_dir="./bert_results",#模型结果保存路径
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=20,  # 课程实际代码用 20（数据少）
            weight_decay=0.01,
            logging_dir="./bert_logs",
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            save_total_limit=1,
            metric_for_best_model="eval_loss",
            fp16=False,
        )

        # =====创建 Trainer 对象 =====
        # 传入：模型、训练参数、训练数据集、验证数据集、评估函数
        # 作用：封装完整的训练循环，调用 trainer.train() 即可自动完成训练
        trainer = Trainer(
            model=self.model,  # 要训练的 BERT 分类模型
            args=training_args,  # 训练参数
            train_dataset=train_dataset,  # 训练数据集
            eval_dataset=val_dataset,  # 验证数据集
            compute_metrics=self.compute_metrics# 评估函数，每个 epoch 结束时会调用，计算准确率
        )
        self.logger.info("开始训练 BERT 模型...")
        trainer.train()

        # 保存模型
        self.save_model()

        #评估模型
        # 调用自定义方法 self.evaluate_model()
        # 传入：val_texts（1000 条验证文本）、val_labels（1000 条验证的数字标签）
        # 作用：用验证集测试模型效果，输出分类报告（precision/recall/f1）和混淆矩阵
        self.evaluate_model(val_texts, val_labels)


    #compute_metrics 是 Trainer 每个 epoch 结束时调用的评估函数，负责计算准确率
    def compute_metrics(self, eval_pred):
        """Trainer 每个 epoch 结束时调用，计算准确率"""
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = (predictions == labels).mean()
        return {"accuracy": accuracy}


    #evaluate_model 是我们自己定义的评估函数，在训练结束后调用，负责用验证集评估模型，输出分类报告和混淆矩阵
    def evaluate_model(self, texts, labels):
        """用验证集评估模型，输出分类报告和混淆矩阵"""
        # ① 只对文本进行分词（labels 已经是数字了，不需要再处理）
        encodings = self.tokenizer(
            texts, truncation=True, padding=True, max_length=128, return_tensors="pt"
        )
        dataset = self.create_dataset(encodings, labels)
        # 指定 output_dir，避免产生默认的 tmp_trainer 文件夹
        eval_args = TrainingArguments(
            output_dir="./bert_results",  # 和训练时共用同一个目录
            per_device_eval_batch_size=8,
        )

        # ② 用 Trainer 做预测
        trainer = Trainer(model=self.model, args=eval_args)
        predictions = trainer.predict(dataset)
        # predictions 是一个 PredictionOutput 对象，包含：
        # predictions.predictions → 模型输出的 logits 数组，shape [N, 2]
        #   例如：[[-4.72, 4.18], [-4.74, 4.21], [3.71, -2.64], ...]
        #          ↑ 通用分数  ↑ 专业分数
        #   第一条：通用 -4.72 < 专业 4.18 → 预测为"专业"(1) ✅
        #   第三条：通用 3.71 > 专业 -2.64 → 预测为"通用"(0) ✅
        # predictions.label_ids → 真实标签数组 [1, 1, 0, 1, 1, 0, ...]
        # predictions.metrics → 评估指标 {'test_loss': 0.0012, ...}

        # ③ 从 logits 中取预测结果
        pred_labels = np.argmax(predictions.predictions, axis=-1)
        # 例如：[[-4.72, 4.18], [3.71, -2.64]] → argmax → [1, 0]
        true_labels = labels  # 直接使用数字标签

        # ④ 输出分类报告
        self.logger.info("分类报告:")
        self.logger.info(classification_report(
            true_labels,#真实标签
            pred_labels,#预测标签
            target_names=["通用知识", "专业咨询"]  # 给数字标签加上文字名称
        ))
        # 输出格式：
        #               precision    recall  f1-score   support
        #      通用知识      1.00      1.00      1.00        25
        #      专业咨询      1.00      1.00      1.00        72
        #     accuracy                           1.00        97

        # ⑤ 输出混淆矩阵
        self.logger.info("混淆矩阵:")
        self.logger.info(confusion_matrix(true_labels, pred_labels))
        # 输出：[[25  0]
        #        [ 0 72]]

    def predict_category(self, query):
        """对用户的一个问题进行分类预测，返回"通用知识"或"专业咨询" """
        # ① 检查模型是否已加载
        if self.model is None:
            self.logger.error("模型未训练或加载")
            return "通用知识"  # 默认返回通用，避免错误地去查知识库

        # ② 对查询进行分词编码
        encoding = self.tokenizer(
            query,  # 单条文本，如 "学费多少"
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"  # 返回 PyTorch 张量
        )
        # ③ 把编码数据搬到模型所在的设备上（CPU 或 GPU）
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        # ④ 推理（不计算梯度，省内存加速）
        with torch.no_grad():
            outputs = self.model(**encoding)
            # **encoding 是 Python 的"字典解包"语法
            # 等价于 self.model(input_ids=..., attention_mask=..., token_type_ids=...)
            # outputs.logits 的形状：[1, 2]
            # 例如：[[-4.72, 4.18]] → 通用 -4.72, 专业 4.18

            # 取分数最高的那个类别的索引
            prediction = torch.argmax(outputs.logits, dim=1).item()
            # argmax → 1（专业分数更高）
            # .item() → 把 PyTorch 张量转成 Python 普通整数
            if prediction == 0:
                return "通用知识"
            else:
                return "专业咨询"





if __name__=="__main__":
    classifier = QueryClassifier()
    print("模型加载成功！")
    classify=classifier.predict_category("Java课程的授课老师")
    print(f"预测结果: {classify}")


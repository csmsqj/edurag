#前置搜索：使用bm25算法进行检索，搜索按照分数排序，返回前k条结果
from base.log import get_logger
from rank_bm25 import BM25L
import numpy as np
import os
import sys
abs_file=os.path.abspath(__file__)
basedir=os.path.dirname(abs_file)
rootdir=os.path.dirname(basedir)
sys.path.insert(0, rootdir)
from utils.test_preProcess import preprocess_text
from mysql.mysql_client import mysql_client
from cache.redis_client import redis_client



class BM25Search:
    # 初始化 BM25Search 类，接受 MySQL 和 Redis 客户端实例

    def __init__(self, mysql_client,redis_client):
        self.logger = get_logger()
        self.mysql_client = mysql_client
        self.redis_client = redis_client
        self.bm25=None
        self.origin_quetions=None
        self.original_questions=None
        self._load_data()


    # 加载数据
    #先从 Redis 获取 → Redis 没有就从 MySQL 加载 → MySQL 有数据则分词并缓存到 Redis → 初始化 BM25 模型。
    def _load_data(self):
        # === 1. 定义 Redis 中的两个键名 ===
        # Redis 中存储了两种数据，各用一个键来标识：
        # 原始问题键：存储未分词的原始问题列表
        # 例如 Redis 中：qa_original_questions → ["什么是机器学习", "Python怎么安装", ...]
        original_key = "qa_original_questions"

        # 分词问题键：存储分词后的问题列表
        # 例如 Redis 中：qa_tokenized_questions → [["什么", "是", "机器", "学习"], ...]
        tokenized_key = "qa_tokenized_questions"

        # === 2. 从 Redis 获取原始问题（速度快） ===
        # self.redis_client.get_data() 是前面 Redis 客户端中写好的方法
        # 它内部执行 cache.get(key)，然后 json.loads() 反序列化为 Python 对象
        # 如果 Redis 中没有这个 key，返回 None
        self.original_questions = self.redis_client.get_data(original_key)

        # === 3. 从 Redis 获取分词后的问题（速度快） ===
        # 同上，获取已经分过词的问题列表
        self.tokenized_questions = self.redis_client.get_data(tokenized_key)

        # === 4. 如果 Redis 中没有数据，则从 MySQL 加载 ===
        # not self.original_questions：原始问题为 None 或空列表
        # not tokenized_questions：分词问题为 None 或空列表
        # 只要有一个没数据，就需要从 MySQL 重新加载
        if not self.original_questions or not self.tokenized_questions:
            # --- 4.1 从 MySQL 获取所有数据 ---
            all_data = self.mysql_client.query_data()
            self.logger.info(f"从 MySQL 加载数据")
            # --- 4.2 如果 MySQL 也没有数据，直接返回 ---
            # 这种情况说明数据库是空的，无法构建任何索引
            if not all_data:
                #先数据库插入数据，再运行程序，才会有数据加载到 Redis 和 BM25 模型中
                # 记录警告日志，方便排查问题
                self.logger.warning("未加载到问题")
                # 直接 return，不初始化 BM25 模型
                # 后续 search() 方法在调用时需要检查 self.bm25 是否为 None
                return

            # --- 4.3 从查询结果中提取原始问题列表 ---
            # query_data() 返回的是 (subject_name, question, answer) 元组
            # 我们只需要 question 字段，即索引 [1] 的位置
            self.original_questions = []
            for q in all_data:
                # q[1] 是问题文本（索引 0 是学科名称，索引 2 是答案）
                self.original_questions.append(q[1])

            # --- 4.4 对原始问题进行分词 ---
            self.tokenized_questions = []
            for question in self.original_questions:
                # 对每个原始问题字符串进行分词
                tokens = preprocess_text(question)
                self.tokenized_questions.append(tokens)
            print(self.tokenized_questions)
            # 等价的列表推导式写法（原代码）：
            # tokenized_questions = [preprocess_text(q[0]) for q in self.original_questions]

            # --- 4.5 根据链式图，将原始数据缓存到 Redis ---
            self.redis_client.set_data(tokenized_key, self.tokenized_questions)
            self.redis_client.set_data(original_key, self.original_questions)

        self.bm25 = BM25L(self.tokenized_questions)

        # === 7. 记录 BM25 初始化成功 ===
        self.logger.info("BM25 模型初始化完成")

    def _softmax(self, scores):
        """Softmax 归一化：将分数转为概率分布"""
        # 减去最大值防止指数溢出（数学上等价）
        exp_scores = np.exp(scores - np.max(scores))
        return exp_scores / exp_scores.sum()

    def search(self, query, threshold=0.6):
        # 输入验证
        if not query or not isinstance(query, str):
            self.logger.error("无效查询")
            return None, False
        # 先查 Redis 缓存
        answer=self.redis_client.get_data("answer:query")
        if answer is not None:
            self.logger.info("从 Redis 缓存获取答案")
            return answer, False
        try:

            #对问题进行分词
            tokenized_query = preprocess_text(query)
            #计算 BM25 分数,获得分数列表
            score_list=self.bm25.get_scores(tokenized_query)
            #对分数列表进行归一化处理，得到概率分布
            scoreone_list=self._softmax(score_list)
            #找到分数最高的索引位置
            #由于是列表形式，所以使用 np.argmax 来找到最大值的索引
            best_index=np.argmax(scoreone_list)
            #根据索引位置获取对应的原始问题
            original_question=self.original_questions[best_index]

            #如果分数最高的那个问题的分数超过了设定的阈值，就返回对应的答案，否则返回 None 和需要人工干预的标志
            if scoreone_list[best_index]>threshold:
                # 根据原始问题，在数据库当中查询对应的答案
                results = self.mysql_client.query_qustion(original_question)
                answer=results[0][2]  # query_qustion 返回的是 (subject_name, question, answer) 元组列表，索引 [0][2] 是第一个结果的答案

                # 将答案缓存到 Redis 中，设置一个过期时间（例如 1 小时）
                self.redis_client.set_data(f"answer:{query}", answer)
                self.logger.info("搜索成功，返回答案")
                return answer, False
            else:
                self.logger.info("搜索结果分数过低，返回需要人工干预的标志")
                return None, True
        except Exception as e:
            self.logger.error(f"搜索失败：{e}")
            return None, True

if __name__ == '__main__':
    logger=get_logger()
    mysql_client = mysql_client()
    redis_client = redis_client()  # 替换为实际的 Redis 客户端实例
    bm25_search = BM25Search(mysql_client, redis_client)
    query = ("导入失败怎么办")
    answer, need_human = bm25_search.search(query)
    if answer is not None:


        print(f"查询: {query}\n答案: {answer}")



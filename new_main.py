# new_main.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mysql_all.cache.redis_client import redis_client
from mysql_all.mysql.mysql_client import mysql_client
from mysql_all.retrieval.bm25_search import BM25Search
from rag.core.vector_store import VectorStore
from rag.core.rag_system import RAGSystem
from base.config import Config
from base.log import get_logger
from langchain_openai import ChatOpenAI    # 用 LangChain 统一管理 LLM
import pymysql
import time
import uuid

config = Config()
logger = get_logger("IntegratedQA")

class IntegratedQA:
    def __init__(self):
        self.redis_client = redis_client()
        self.mysql_client = mysql_client()
        self.bm25_search = BM25Search(self.mysql_client, self.redis_client)
        self.llm = ChatOpenAI(
            model=config.MODEL,
            base_url=config.URL,
            api_key=config.DEEPSEEK_APIKEY,
            temperature=0.2,
            streaming=True
        )

        self.rag_system = RAGSystem(self.llm)
        # 建对话历史表
        self.init_conversation_table()
        self.max_length = 8000


#创建·对话历史表
    def init_conversation_table(self):
        """在 MySQL 创建 conversations 表（幂等，程序每次启动都跑）"""
        try:
            self.mysql_client.cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id          INT AUTO_INCREMENT PRIMARY KEY,
                session_id  VARCHAR(36) NOT NULL,
                question    TEXT NOT NULL,
                answer      TEXT NOT NULL,
                timestamp   DATETIME NOT NULL,
                INDEX idx_session_id (session_id)
            )
        """)
            self.mysql_client.connection.commit()
            logger.info("对话历史表初始化成功")
        except pymysql.MySQLError as e:
            logger.error(f"初始化对话历史表失败: {e}")
            raise


    #查询过去5条对话历史
    def get_conversation_history(self, session_id):
        """查询指定 session_id 的最近 5 条对话历史"""
        try:
            self.mysql_client.cursor.execute("""
                SELECT question, answer, timestamp
                FROM conversations
                WHERE session_id = %s
                ORDER BY timestamp DESC
                LIMIT %s
            """, (session_id,5))
            rows = self.mysql_client.cursor.fetchall()
            # 将结果按时间顺序返回（最早的在前）
            return rows[::-1]
        except pymysql.MySQLError as e:
            logger.error(f"查询对话历史失败: {e}")
            return []


    #插入对话历史，注意：这里是先插入，在只保留最后5条信息（先查询在删除，这种情况必须把子查询）
    def insert_conversation_history(self, session_id, question, answer):
        """插入一条对话历史，并确保每个 session_id 只保留最近 5 条记录"""
        try:
            # 插入新记录
            self.mysql_client.cursor.execute("""
                INSERT INTO conversations (session_id, question, answer, timestamp)
                VALUES (%s, %s, %s, NOW())
            """, (session_id, question, answer))

            # 删除超过 5 条的旧记录
            self.mysql_client.cursor.execute("""
                DELETE FROM conversations
                WHERE id NOT IN (
                    SELECT id FROM (
                        SELECT id
                        FROM conversations
                        WHERE session_id = %s
                        ORDER BY timestamp DESC
                        LIMIT 5
                    ) AS sub
                ) AND session_id = %s
            """, (session_id, session_id))
            self.mysql_client.connection.commit()
        except pymysql.MySQLError as e:
            logger.error(f"插入对话历史失败: {e}")
            self.mysql_client.connection.rollback()  # 回滚事务，保持数据一致性
            raise



    #结合总项目的入口
    def answer_question(self, question, session_id=None):
        #记录开始时间
        start_time = time.time()
        if not session_id:
            session_id = str(uuid.uuid4())  # 如果没有提供 session_id，则生成一个新的 UUID
        logger.info(f"开始处理查询: {question} (session_id: {session_id})")
        # 查询过去的对话历史，数据库返回元素为元组，组成一个元组
        conversation_history = self.get_conversation_history(session_id)
        #1查询：先bm25检索，获取相关文档
        bm25_results,is_rag = self.bm25_search.search(question,0.7)
        if bm25_results is not None:
            # 【分支 A】FAQ 命中 → 一次性返回完整答案
            self.insert_conversation_history(session_id,question, bm25_results)
            #流式输出的方法是使用 YIELD 输出这个返回这个答案。第2个参数表示是否结束
            yield bm25_results, True
            return # 直接返回，不再继续执行后续代码
        if is_rag:
            # 【分支 B】RAG 检索 → 生成答案
            rag_results=""
            for chunk in self.rag_system.generate_answer(question,history=conversation_history):
                rag_results+=chunk
                yield chunk, False  # 流式输出每个 chunk，False 表示还没结束

            self.insert_conversation_history(session_id,question, rag_results)
            yield "", True  # 最后输出一个空字符串，True 表示结束
            return
        else:
            # 【分支 C】无效查询
            default_response = "抱歉，我无法回答您的问题。请尝试重新表述您的问题，或者联系人工客服。"
            self.insert_conversation_history(session_id,question, default_response)
            yield default_response,True
            return

if __name__ == "__main__":
    qa_system = IntegratedQA()
    session_id = str(uuid.uuid4())  # 生成一个新的 session_id
    question = "HNSW 图索引的邻居选择策略对召回率有什么影响"
    answer=qa_system.answer_question(question, session_id=session_id)
    for chunk, is_end in answer:
        print(chunk, end='', flush=True)
        if is_end:
            print("\n[回答结束]")



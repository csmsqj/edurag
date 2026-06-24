
import time           # 计时模块，记录处理耗时
import os
import sys
rootfile=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
dirfile=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, dirfile)   # 将当前目录添加到 sys.path，确保可以导入 strategy_selector 模块
sys.path.insert(0, rootfile)  # 将项目根目录添加到 sys.path
from base.config import Config          # 日志 + 配置
from base.log import get_logger
from base.output_parser import ThinkingStripper
from prompts import RAGPrompts  # 查询分类器
from strategy_selector import Strategy_selector  # 策略选择器（本笔记第六章）
from langchain_openai import ChatOpenAI         # LangChain 封装的大模型调用类
from query_classifier import QueryClassifier
from vector_store import VectorStore  # 向量数据库类，提供混合检索+重排序方法
config=Config()
class RAGSystem:
    #初始化通用的 RAG 系统组件
    def __init__(self, llm):
        self.llm = llm
        self.logger = get_logger("RAGSystem")
        self.stripper = ThinkingStripper()
        #直接提示词模板，适用于所有需要生成答案的场景（通用知识和专业咨询）
        self.rag_prompt = RAGPrompts.rag_prompt()
        self.rag_chain = self.rag_prompt | self.llm | self.stripper
        self.query_classifier = QueryClassifier()
        self.strategy_selector = Strategy_selector()#这是策略选择器，负责根据用户查询选择最合适的检索增强策略
        self.vector_store = VectorStore()#这是向量数据库类，提供混合检索+重排序方法，用于检索相关文档块
#试用llm生成假设答案的检索策略
    def _retrieve_with_hyde(self, query):
        """使用 HyDE 策略：先生成假设答案，再用假设答案去检索"""
        self.logger.info(f"使用 HyDE 策略进行检索（查询: '{query}')")

        # ① 获取 HyDE 的 Prompt 模板（ChatPromptTemplate 对象）
        hyde_prompt_template = RAGPrompts.hyde_prompt()

        try:
            hyde_chain = hyde_prompt_template | self.llm | self.stripper
            hypo_answer = hyde_chain.invoke({"query": query})
            self.logger.info(f"HyDE 生成的假设答案: '{hypo_answer}'")
            # 例如：query = "人工智能在教育领域有哪些应用"
            #       hypo_answer = "人工智能在教育领域的应用包括智能辅导、自动批改..."

            # ③ 用假设答案（而非原始问题）去向量库检索
            return self.vector_store.search(
                hypo_answer  # ⬅️ 注意：用的是假设答案，不是原始问题
            )
        except Exception as e:
            self.logger.error(f"HyDE 策略执行失败: {e}")
            return []
#使用llm生成子查询的检索策略(对问题进行处理并且检索)
    def _retrieve_with_subqueries(self, query):
        """使用子查询策略：把复杂问题拆成多个简单子问题，分别检索再合并"""
        self.logger.info(f"使用子查询策略进行检索（查询: '{query}')")

        subquery_prompt_template = RAGPrompts.subquery_prompt()

        try:
            subquery_chain = subquery_prompt_template | self.llm | self.stripper
            subqueries_text = subquery_chain.invoke({"query": query})
            # 例如：query = "对比 AI 课程和 Java 课程"
            #       subqueries_text = "AI 课程的内容是什么\nJava 课程的内容是什么"

            # ② 按换行符拆成子查询列表
            subqueries = [q.strip() for q in subqueries_text.split("\n") if q.strip()]

            # ③ 对每个子查询分别检索
            all_results = []
            for sub_query in subqueries:
                results = self.vector_store.search(
                    sub_query
                )
                all_results.extend(results)


            # ⚠️ 逻辑漏洞：这里没有对 all_results 做去重处理。
            # 如果两个子查询检索到了同一个文档块，all_results 中会出现重复内容，
            # 后面 generate_answer 拼接上下文时会把同一段文本重复拼入 Prompt，
            # 浪费 token 且可能干扰大模型回答。
            # 改进方案：可以根据文档 ID 或 page_content 去重，例如：
            seen = set()
            unique_results = []
            for doc in all_results:
                if doc.page_content not in seen:
                   seen.add(doc.page_content)
                   unique_results.append(doc)
            return unique_results

        except Exception as e:
            self.logger.error(f"子查询策略执行失败: {e}")
            return []


#使用llm生成回溯问题的检索策略，返回检索的结果
    def _retrieve_with_backtracking(self, query):
        """使用回溯策略：把复杂问题简化为更基础的问题，用简化版检索"""
        self.logger.info(f"使用回溯问题策略进行检索（查询: '{query}')")

        backtrack_prompt_template = RAGPrompts.backtracking_prompt()

        try:
            backtrack_chain = backtrack_prompt_template | self.llm | self.stripper
            simplified_query = backtrack_chain.invoke({"query": query})
            self.logger.info(f"生成的回溯问题: '{simplified_query}'")
            # 例如：query = "HNSW 图索引的邻居选择策略对召回率有什么影响"
            #       simplified_query = "什么是 HNSW 索引"

            # 用简化后的问题去检索
            return self.vector_store.search(
                simplified_query
            )
        except Exception as e:
            self.logger.error(f"回溯问题策略执行失败: {e}")
            return []

#根据传入的检索策略进行检索,返回检索的结果
    def retrieve_and_merge(self, query, strategy=None):
        """根据检索策略选择对应的检索方法"""

        # 如果没有指定策略，用策略选择器自动选
        if not strategy:
            strategy = self.strategy_selector.get_strategy(query)



        # 根据策略名称，调用对应的检索方法
        ranked_sub_chunks = []
        if strategy == "回溯问题检索":
            ranked_sub_chunks = self._retrieve_with_backtracking(query)
        elif strategy == "子查询检索":
            ranked_sub_chunks = self._retrieve_with_subqueries(query)
        elif strategy == "假设问题检索":
            ranked_sub_chunks = self._retrieve_with_hyde(query)
        else:  # 默认：直接检索
            self.logger.info(f"使用直接检索策略（查询: '{query}')")
            ranked_sub_chunks = self.vector_store.search(
                query
            )

        return ranked_sub_chunks

# RAG 系统的核心方法：从用户提问到生成最终答案的完整流程
    def generate_answer(self, query):
        """从用户提问到生成最终答案的完整流程"""
        # ===== 第一步：记录开始时间 =====
        start_time = time.time()
        self.logger.info(f"开始处理查询: {query}")
        # ===== 第二步：用 BERT 分类器判断问题类型 =====
        query_category = self.query_classifier.predict_category(query)
        self.logger.info(f"查询分类结果: {query_category} (查询: '{query}')")

        # ===== 第三步：分支处理 =====
        # 【分支 A】通用知识 → 直接让大模型回答，不走知识库
        if query_category == "通用知识":
            self.logger.info("查询为通用知识，直接调用 LLM")
            try:
                result = self.rag_chain.invoke({
                    "context": "",
                    "question": query,
                    "phone": 13339833311
                })
                answer = result
                # result 是 AIMessage 对象，result.content 是大模型返回的文本
            except Exception as e:
                self.logger.error(f"直接调用 LLM 失败: {e}")
                answer = f"抱歉，处理您的通用知识问题时出错。请联系人工客服：{13339833311}。"

            processing_time = time.time()- start_time
            self.logger.info(f"通用知识查询处理完成 (耗时: {processing_time:.2f}s, 查询: '{query}')")
            return answer

        # 【分支 B】专业咨询 → 走 RAG 流程
        self.logger.info("查询为专业咨询, 执行 RAG 流程")

        # ① 返回检索策略
        strategy = self.strategy_selector.get_strategy(query)

        # ② 根据策略检索相关文档
        # source_filter 会被传递到 retrieve_and_merge → hybrid_search_with_rerank
        # 作用：只从指定学科的文档中检索（如 source_filter="ai" 则只搜 AI 相关文档）
        # 如果 source_filter=None，则搜索所有学科的文档（不过滤）
        context_docs = self.retrieve_and_merge(
            query, strategy=strategy
        )

        # ③ 拼接上下文（多个文档用换行符分隔）
        # context_docs 是 Document 对象列表，每个 doc 有 page_content 属性（文本内容）
        if context_docs:
            context = "\n\n".join([doc.page_content for doc in context_docs])
            # 使用 "\n\n"（两个换行）分隔各文档块，让大模型能区分不同来源的信息
            self.logger.info(f"构建上下文完成, 包含 {len(context_docs)} 个文档块")
        else:
            context = ""
            self.logger.info("未检索到相关文档, 上下文为空")

        # ④ 最终使用 LCEL 链式调用生成答案

        result = self.rag_chain.invoke({
            "context": context,
            "question": query,
            "phone": 13339833311
        })
        # invoke 内部执行流程：
        # 1. rag_prompt.invoke({"context": ..., "question": ..., "phone": ...})
        #    → 生成 [SystemMessage("你是一个智能助手..."), HumanMessage("上下文：...问题：...")]
        # 2. llm.invoke(上一步的消息列表)
        #    → 发送给 DeepSeek API，返回 AIMessage(content="AI大模型班学费25800元...")


        return result



if __name__=="__main__":
    llm=ChatOpenAI(model=config.MODEL,
                     base_url=config.URL,
                        api_key=config.DEEPSEEK_APIKEY
                   )

    rag_system=RAGSystem(llm)
    answer=rag_system.generate_answer("Java课程的授课老师是谁？")
    print(answer)




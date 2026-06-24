# core/prompts.py
#四种策略选择策略的处理方法的 Prompt 模板集中管理类
from langchain_core.prompts import ChatPromptTemplate
class RAGPrompts:
    """集中管理所有 Prompt 模板的类"""

    # ───────────────────────────────────
    # ① 核心回答 Prompt（所有版本都用，最重要的一个）
    # 被 rag_system.py 的 generate_answer() 方法调用
    # ───────────────────────────────────
    @staticmethod
    def rag_prompt():
        return ChatPromptTemplate.from_messages([
            ("system", "你是一个智能助手，帮助用户回答问题。"
                       "如果提供了上下文，请基于上下文回答；如果没有上下文，请直接根据你的知识回答。"
                       "如果答案来源于检索到的文档，请在回答中说明。"
                       "最好不要的行为：如果完全无法无法回答，才可以请回复：\"信息不足，无法回答，请联系人工客服，电话：{phone}。\""),
            ("user", "上下文：{context}\n\n问题：{question}\n\n回答：")
        ])
        # 变量说明：
        # context  → 检索到的文档内容（父块文本），由 hybrid_search_with_rerank 返回
        # question → 用户的原始问题
        # phone    → 客服电话号码，从 config.ini 读取

    # ───────────────────────────────────
    # ② HyDE（假设性文档嵌入）Prompt
    # 被 rag_system.py 的 _retrieve_with_hyde() 方法调用
    # 思路：用户问"什么是好的代码"，直接检索可能效果差
    # 先让 LLM 生成一个假设答案"好的代码应该具备可读性、可维护性..."
    # 然后用这个假设答案去向量库检索，匹配效果更好
    # ───────────────────────────────────
    @staticmethod
    def hyde_prompt():
        return ChatPromptTemplate.from_messages([
            ("system", "假设你是用户，想了解以下问题，请生成一个简短的假设答案。"),
            ("user", "问题：{query}\n假设答案：")
        ])

    # ───────────────────────────────────
    # ③ 子查询 Prompt
    # 被 rag_system.py 的 _retrieve_with_subqueries() 方法调用
    # 思路：把复杂问题拆成多个简单子问题，分别检索再合并结果
    # 例如："对比 Redis 和 MySQL 的持久化机制" →
    #   子查询1："Redis 的持久化机制是什么"
    #   子查询2："MySQL 的持久化机制是什么"
    # ───────────────────────────────────
    @staticmethod
    def subquery_prompt():
        return ChatPromptTemplate.from_messages([
            ("system", "你是一个智能助手，负责将复杂查询分解为多个简单子查询。"),
            ("user", "将以下复杂查询分解为多个简单子查询，每行一个子查询：\n查询：{query}\n子查询：")
        ])

    # ───────────────────────────────────
    # ④ 回溯简化 Prompt
    # 被 rag_system.py 的 _retrieve_with_backtracking() 方法调用
    # 思路：把过于复杂的问题简化为更基础的问题
    # 例如："HNSW 图索引的邻居选择策略对召回率的影响" → "什么是 HNSW 索引"
    # ───────────────────────────────────
    @staticmethod
    def backtracking_prompt():
        return ChatPromptTemplate.from_messages([
            ("system", "你是一个智能助手，负责将复杂查询简化为更基础、更易于检索的问题。"),
            ("user", "将以下复杂查询简化为一个更简单的问题：\n查询：{query}\n简化问题：")
        ])
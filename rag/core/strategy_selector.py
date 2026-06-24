from langchain_core.prompts import ChatPromptTemplate  # LangChain 的 Chat 模板类
from langchain_openai import ChatOpenAI       # LangChain 封装的 OpenAI 兼容客户端
import os
import sys
rootfile=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, rootfile)  # 将项目根目录添加到 sys.path，确保可以导入 base 模块
# 导入日志和配置
from base.config import Config
from base.log import get_logger
from base.output_parser import ThinkingStripper

class Strategy_selector:
    def __init__(self):
        self.config = Config()
        self.logger = get_logger("Strategy_selector")
        self.llm = ChatOpenAI(model=self.config.MODEL,
                              base_url=self.config.URL,
                              api_key=self.config.DEEPSEEK_APIKEY,
                              temperature=0.1
                              )
        self.stripper = ThinkingStripper()
        self.strategy_prompt=self._get_strategy_prompt()
        self.strategy_chain=self.strategy_prompt | self.llm | self.stripper

   #作用是返回策略选择器提示词
    def _get_strategy_prompt(self):
        """定义策略选择的 Prompt 模板（使用 ChatPromptTemplate）"""
        return ChatPromptTemplate.from_messages([
            ("system", "你是一个智能助手，负责分析用户查询并选择最合适的检索增强策略。"
                       "直接返回策略名称，不要输出任何分析过程或其他内容。"),
            ("user", """\
分析用户查询 {query}，并从以下四种检索增强策略中选择一个最适合的。以下是几种检索增强策略及其适用场景：
1. **直接检索：**
   * 描述：对用户查询直接进行检索，不进行任何增强处理。
   * 适用场景：查询意图明确，需要从知识库中检索特定信息。
   * 示例：
     * "AI 学科学费是多少？" → 直接检索
     * "JAVA的课程大纲是什么？" → 直接检索

2. **假设问题检索 (HyDE)：**
   * 描述：使用 LLM 生成一个假设的答案，然后基于假设答案进行检索。
   * 适用场景：查询较为抽象，直接检索效果不佳。
   * 示例：
     * "云计算课程有没有针对零基础学员的" → 假设问题检索

3. **子查询检索：**
   * 描述：将复杂的用户查询拆分为多个简单子查询，分别检索并合并结果。
   * 适用场景：查询涉及多个实体或方面。
   * 示例：
     * "对比 AI 课程和 Java 课程的区别" → 子查询检索

4. **回溯问题检索：**
   * 描述：将复杂查询转化为更基础、更易于检索的问题。
   * 适用场景：查询较为复杂，需要简化后才能有效检索。
   * 示例：
     * "我有 100 亿条记录想存到 Milvus，可以吗？" → 回溯问题检索

根据用户查询 {query}，直接返回最适合的策略名称，例如 "直接检索"。""")
        ])


    #返回策略选择器的结果
    def get_strategy(self, query):
        """根据用户查询返回策略选择结果"""
        return self.strategy_chain.invoke({"query": query})







if __name__ == '__main__':
    strategy_selector=Strategy_selector()
    result=strategy_selector.get_strategy("我有 100 亿条记录想存到 Milvus，可以吗？")
    print(result)




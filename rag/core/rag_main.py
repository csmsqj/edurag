#普通知识和rag知识的问答系统入口文件（没有数据库bm25）
# ===== 标准库 =====
import os       # 文件和目录操作（os.path.exists、os.path.join 等）
import sys      # 系统相关操作（本文件中未直接使用，但保留以备扩展）

import argparse

rootfile=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, rootfile)  # 将项目根目录添加到 sys.path，确保可以导入 base 模块
dirfile=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, dirfile)   # 将当前目录添加到 sys.path
# ===项目内部模块====
from base.config import Config                           # 配置类
from base.log import get_logger                          # 日志函数
from rag.core.document_loader_splitter import document_loader_splitter  # 导入文档加载切分类
# process_documents 的作用：读取指定目录下的文档文件 → 切分成 chunks → 返回 chunk 列表
from rag.core.vector_store import VectorStore                # 向量存储类
# VectorStore 提供：add_documnets（存入向量库）、search（混合检索+重排）

from rag.core.rag_system import RAGSystem                    # RAG 核心系统
# RAGSystem 串联了：查询分类 → 策略选择 → 检索 → 生成回答（本笔记第七章已详细讲解）

# ===== 第三方库 =====
from langchain_openai import ChatOpenAI  # LangChain 封装的 OpenAI 兼容客户端
# ChatOpenAI 内部封装了 OpenAI SDK 的全部调用逻辑：
config=Config()
logger=get_logger()

def main():
    ...# ===== 标准库 =====
import os       # 文件和目录操作（os.path.exists、os.path.join 等）
import sys      # 系统相关操作（本文件中未直接使用，但保留以备扩展）
rootfile=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, rootfile)  # 将项目根目录添加到 sys.path，确保可以导入 base 模块
dirfile=os.path.dirname(os.path.abspath(__file__))
print(f"当前文件路径: {dirfile}")
sys.path.insert(0, dirfile)   # 将当前目录添加到 sys.path
# ===项目内部模块====
from base.config import Config                           # 配置类
from base.log import get_logger                          # 日志函数
from core.document_loader_splitter import document_loader_splitter  # 导入文档加载切分类
# process_documents 的作用：读取指定目录下的文档文件 → 切分成 chunks → 返回 chunk 列表
from core.vector_store import VectorStore                # 向量存储类
# VectorStore 提供：add_documnets（存入向量库）、search（混合检索+重排）

from core.rag_system import RAGSystem                    # RAG 核心系统


# ===== 第三方库 =====
from langchain_openai import ChatOpenAI  # LangChain 封装的 OpenAI 兼容客户端
# ChatOpenAI 内部封装了 OpenAI SDK 的全部调用逻辑：
config=Config()
logger=get_logger()
file=os.path.join(rootfile,"data")
def main(query_mode=True, directory_path=file):
    logger.info("RAG 系统启动中...")
    # 1初始化大模型
    llm = ChatOpenAI(model=config.MODEL,
                     base_url=config.URL,
                     api_key=config.DEEPSEEK_APIKEY,
                     temperature=0.2
                     )
    #2初始化milvus向量存储
    vector_store = VectorStore()
    #3根据命令行判断是加载文件到milvus，还是启动RAG问答系统
    if query_mode:
        # 初始化RAG系统
        rag_system = RAGSystem(llm)
        while True:
            user_query = input("请输入您的问题（输入 'exit' 退出）：")
            if user_query.lower() == 'exit':
                print("退出问答系统。")
                break
            result=rag_system.generate_answer(user_query)
            print(f"RAG 系统回答：{result}")
    else:
        # 加载文件到向量存储
        loader_splitter = document_loader_splitter()
        documents = loader_splitter.load_documents_from_directory(directory_path)
        vector_store.add_documents(documents)
        logger.info("文档加载完成。")


if __name__=="__main__":
    """
        命令行入口：使用 argparse 解析参数，然后调用 main() 函数。
    """
    # ① 创建参数解析器
    parser = argparse.ArgumentParser(description='EduRAG 智慧问答系统运行入口')
    # ② 添加参数：--query_mode，指定是否进入问答模式（默认 True）
    parser.add_argument('--query_mode', default=True, help='是否进入交互式查询模式（不带此参数则默认进入数据处理模式）')
    # ③ 添加参数：--directory_path，指定文档目录路径（默认 "./data"）
    parser.add_argument('--directory_path', type=str, default=file, help='文档目录路径')
    # ③ 解析命令行参数
    args = parser.parse_args()
    # ④ 调用 main() 函数，传入解析后的参数
    main(query_mode=args.query_mode, directory_path=args.directory_path)





import os
import sys
absfile=os.path.abspath(__file__)
basedir=os.path.dirname(absfile)
rootdir=os.path.dirname(basedir)
dir=os.path.dirname(rootdir)
sys.path.insert(0,dir)
from base.config import Config
from base.log import get_logger
#虽然在前面已经写了图片加载等等，但是对于 TXT 加载和 MD 文档加载，直接用官方提供了即可。对于 MD 文档的切割也直接用官方提供的切割即可
# 功能：加载纯文本文件（.txt），把文件内容读成一个 Document 对象
from langchain_community.document_loaders import TextLoader
# 功能：加载 Markdown 文件（.md），能解析 Markdown 的标题、列表、代码块等结构
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain_community.document_loaders import CSVLoader
from edu_document_loaders.edu_pdfloader import OCRPDFLoader
from edu_document_loaders.edu_docloader import CustomDocLoader
from edu_document_loaders.edu_pptloader import CustomPPTLoader
from edu_document_loaders.edu_imgloader import CustomImageLoader

# 注意：LangChain 0.2+ 版本已把 text_splitter 拆分到独立包 langchain-text-splitters
#       旧写法 from langchain.text_splitter import ... 在新版中会报 ImportError
from langchain_text_splitters import MarkdownTextSplitter
from edu_text_splitter.edu_chinese_text_splitter import ChineseRecursiveTextSplitter



from datetime import datetime  # Python 内置模块，用于记录处理时间

# 定义支持的文件类型及其对应的加载器字典
document_loaders = {
    ".txt": TextLoader,
    ".pdf": OCRPDFLoader,
    ".doc": CustomDocLoader,
    ".docx": CustomDocLoader,
    ".ppt": CustomPPTLoader,
    ".pptx": CustomPPTLoader,
    ".jpg": CustomImageLoader,
    ".png": CustomImageLoader,
    ".md": UnstructuredMarkdownLoader,
    ".csv":CSVLoader
}


class document_loader_splitter:
    def __init__(self):
       self.config=Config()
       self.logger=get_logger()
    # 从指定目录加载所有支持格式的文件，返回一个 Document 对象列表
    def load_documents_from_directory(self,directory_path):
        """从指定目录加载所有支持格式的文件"""
        documents = []  # 创建空列表，用来收集所有加载出来的 Document 对象
        # os.listdir(目录路径) —— Python 内置方法
        # 功能：列出指定目录下的所有文件名和子目录名，返回一个字符串列表
        # 例如：os.listdir('./data') → ['LLM基础知识.pdf', '课程大纲.md', 'logo.png']
        # 注意：只返回文件名，不包含完整路径
        for filename in os.listdir(directory_path):
            filepath=os.path.join(directory_path,filename)  # 拼接成完整路径
            ext=os.path.splitext(filename)[1].lower()  # 获取文件扩展名并
            print(f"正在处理文件: {filename}，扩展名: {ext}")
            if ext in document_loaders:  # 判断扩展名是否在支持的加载器字典中
                try:
                    loader_class=document_loaders[ext]  # 获取对应的加载器类
                    if ext in ['.txt','.md','.csv']:
                        loader=loader_class(filepath,encoding='utf-8')  # 实例化加载器对象，传入文件路径和编码
                    else:
                        loader=loader_class(filepath)  # 实例化加载器对象，传入文件路径
                    loader_documents=list(loader.lazy_load()) # 调用加载器的 lazy_load() 方法，返回一个 Document 对象列表
                    # extend() 把 docs 列表中的每个元素追加到 documents 列表中
                    # 注意和 append() 的区别：
                    #   append([1,2,3]) → [[1,2,3]]  （把整个列表当作一个元素追加）
                    #   extend([1,2,3]) → [1,2,3]    （把列表中的每个元素逐个追加）
                    documents.extend(loader_documents)
                    self.logger.info(f"成功加载文件: {filename},共 {len(loader_documents)} 个 Document 对象")
                except Exception as e:
                    self.logger.error(f"加载文件失败: {filename}，错误信息: {str(e)}")
            else:
                self.logger.warning(f"不支持的文件类型: {filename}，跳过处理")
        return documents
    # 对加载出来的 Document 对象进行切割，返回一个新的 Document 对象列表
    def process_documents(self,directory_path, parent_chunk_size=None,child_chunk_size=None,chunk_overlap=None):
        if parent_chunk_size is None:
            parent_chunk_size=self.config.PARENT_CHUNK_SIZE
        if child_chunk_size is None:
            child_chunk_size=self.config.CHILD_CHUNK_SIZE
        if chunk_overlap is None:
            chunk_overlap=self.config.CHUNK_OVERLAP
        #调用 load_documents_from_directory() 方法加载文档
        documents=self.load_documents_from_directory(directory_path)
        self.logger.info(f"加载的文档数量：{len(documents)}")
        #创建切割器分为 MD 文档切割器和中文文档切割器，同时还分了父切割器和子切割器
        # parent_md_size
        md_splitter=MarkdownTextSplitter(chunk_size=parent_chunk_size,chunk_overlap=chunk_overlap)
        #parent_chinese_size
        chinese_splitter=ChineseRecursiveTextSplitter(chunk_size=parent_chunk_size,chunk_overlap=chunk_overlap)
        # child_md_size
        md_child_splitter=MarkdownTextSplitter(chunk_size=child_chunk_size,chunk_overlap=chunk_overlap)
        # child_chinese_size
        chinese_child_splitter=ChineseRecursiveTextSplitter(chunk_size=child_chunk_size,chunk_overlap=chunk_overlap)
        all_chunks=[]  # 创建空列表，用来收集所有切割出来的 Document 对象
        #  i = 当前文档的编号（从 0 开始），doc = 当前文档的 Document 对象
        # 为什么能用两个变量接收？Python 支持"元组解包"——(i, doc) 自动拆成 i 和 doc
        # 为什么需要编号 i？因为后面要给每个块生成唯一 ID，ID 里需要包含文档编号
        for i, doc in enumerate(documents):
            #metadata={"source": self.file_path, "page": page_num}
            filepath=doc.metadata["source"]
            ext=os.path.splitext(filepath)[1].lower()  # 获取文件扩展名并转换为小写
            if ext == '.md':  # 如果是 Markdown 文档，使用 md_splitter 切割
                parent_chunks=md_splitter
                child_chunks=chinese_splitter
            else:  # 对于其他类型的文档，使用 chinese_splitter 切割
                parent_chunks=md_child_splitter
                child_chunks=chinese_child_splitter
            # 调用父切割器的 split_text() 方法，传入文档内容
            parent_docs = parent_chunks.split_documents([doc])
            # parent_docs是一个 Document 对象列表，是一个原本页切割出来的多个块，包含 page_content 和 metadata（对每一个块在进行切割）
            for j, parent_doc in enumerate(parent_docs):
                # 调用子切割器的 split_text() 方法，传入父块的内容
                child_docs = child_chunks.split_documents([parent_doc])
                # child_docs是一个 Document 对象列表，是一个父块切割出来的多个小块，包含 page_content 和 metadata（对每一个小块在进行切割）
                for k, child_doc in enumerate(child_docs):
                    #这里遍历子框要加入对应的 ID 等元素，然后再把子框加入到列表当中去
                    parent_id=f"doc{i}_parent{j}"  # 父块 ID，格式为 doc0_parent0、doc0_parent1、doc1_parent2 等
                    child_doc.metadata["parent_id"]=parent_id  # 把父块 ID 存到 metadata 字典里，键名为 "parent_id"
                    child_doc.metadata["parent_content"]=parent_doc.page_content  # 把父块内容存到 metadata 字典里，键名为 "parent_content"
                    all_chunks.append(child_doc)  # 把切割出来的子块 Document 对象添加到 all_chunks 列表中



        self.logger.info(f"切割后的文档块数量：{len(all_chunks)}")
        return all_chunks  # 返回切割后的所有块的 Document 对象列表

if __name__ == '__main__':
    loader_splitter=document_loader_splitter()
    directory_path=os.path.join(dir,'data')
    #要求传的是具体对应的目录路径，例如：D:/myfiles/data 等等，不能传根目录或者不存在的目录，否则会报错
    loader_splitter.process_documents(directory_path)











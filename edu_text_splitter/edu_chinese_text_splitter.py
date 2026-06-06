from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
class ChineseRecursiveTextSplitter(RecursiveCharacterTextSplitter):
    """针对中文优化的递归文本切分器"""

    def __init__(self, chunk_size=192, chunk_overlap=50, **kwargs):
        # 中文分隔符优先级：段落 > 句号 > 问号 > 感叹号 > 分号 > 逗号 > 空格
        separators = [
            "\n\n",
            "\n",
            "。",
            "？",    # ← 这就是"可以加问号"的意思
            "！",
            "；",
            "，",
            " ",
            ""
        ]
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            **kwargs
        )

if __name__ == "__main__":
    import sys
    import os
    file=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, file)
    fileload=os.path.join(file,"data/test.png")
    from edu_document_loaders.edu_imgloader import CustomImageLoader
    # 加载文档
    docs=CustomImageLoader(fileload).lazy_load()
    print(type(docs))
    # 通用切分器（父块和子块各一个）
    parent_splitter = ChineseRecursiveTextSplitter(
        chunk_size=512,
        chunk_overlap=50
    )
    child_splitter = ChineseRecursiveTextSplitter(
        chunk_size=128,
        chunk_overlap=50
    )
    for doc in docs:
        print(f"原文长度: {len(doc.page_content)}")
        parent_documents=parent_splitter.split_documents([doc])
        print(f"父块数量: {len(parent_documents)}")
        for i, parent_doc in enumerate(parent_documents):
            print(f"父块第{i} 长度: {len(parent_doc.page_content)}")
            child_documents=child_splitter.split_documents([parent_doc])
            print(f"  子块数量: {len(child_documents)}")



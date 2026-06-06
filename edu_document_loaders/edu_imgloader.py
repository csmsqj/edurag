from typing import Iterator
import os
import sys
absfile = os.path.abspath(__file__)
dirfile = os.path.dirname(absfile)
sys.path.insert(0, dirfile)  # 将当前文件所在目录加入 sys.path
from edu_ocr import get_ocr
from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader

class CustomImageLoader(BaseLoader):
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path

    def lazy_load(self) -> Iterator[Document]:
        # 直接传文件路径给 get_ocr（RapidOCR 支持 str 路径输入）
        text = get_ocr(self.file_path)
        if text.strip():
            yield Document(
                page_content=text,
                metadata={"source": self.file_path}
            )
if __name__ == '__main__':


    absfile = os.path.abspath(__file__)
    dirfile = os.path.dirname(absfile)
    dirfile = os.path.dirname(dirfile)
    file = os.path.join(dirfile, 'data/test.png')
    loader = CustomImageLoader(file)
    docs = loader.load()
    print(f'识别出的文字: {docs[0].page_content[:100] if docs else "无内容"}')
    # 如果输出了图片中的文字 → 图片加载器没问题
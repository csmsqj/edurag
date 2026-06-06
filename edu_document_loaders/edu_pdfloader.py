from typing import Iterator
import os
absfile=os.path.abspath(__file__)
dirfile=os.path.dirname(absfile)
import sys
sys.path.insert(0, dirfile)  # 将当前文件所在目录加入 sys.path，确保能导入 edu_ocr
from edu_ocr import get_ocr
from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader


#使用方法初始化时传对应的路径，然后调用加载方法返回对应的document列表
class OCRPDFLoader(BaseLoader):
    """支持 OCR 的 PDF 加载器"""

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path

    def lazy_load(self) -> Iterator[Document]:
        # 用 fitz (PyMuPDF) 打开 PDF
        import fitz
        doc = fitz.open(self.file_path)

        for page_num, page in enumerate(doc):
            # 先尝试直接提取文字
            text = page.get_text()

            # 如果文字太少，尝试 OCR
            if len(text.strip()) < 50:
                # 把页面渲染成图片
                pix = page.get_pixmap()
                img = pix.tobytes("png")
                # 调用 OCR
                text = get_ocr(img)

            if text.strip():
                yield Document(
                    page_content=text,
                    metadata={"source": self.file_path, "page": page_num}
                )

if __name__ == '__main__':
    import os
    absfile = os.path.abspath(__file__)
    dirfile = os.path.dirname(absfile)
    dirfile = os.path.dirname(dirfile)
    file = os.path.join(dirfile, 'data/LLM基础知识.pdf')
    loader = OCRPDFLoader(file)
    document=loader.lazy_load()
    for doc in document:
        print(f"page_content->{doc.page_content[:50]}")  # 打印每页前10个字符，验证是否成功加载文本
        print(f"{doc.metadata.get('page')}")  # 打印元数据，验证是否正确记录页码和来源

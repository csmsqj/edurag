# edu_ocr.py 核心逻辑
from rapidocr_onnxruntime import RapidOCR

def get_ocr(image) -> str:
    """
    输入：一张图片（PIL Image 或 numpy array）
    输出：识别出的全部文字拼接成的字符串
    """
    ocr_engine = RapidOCR()
    result, _ = ocr_engine(image)
    if result:
        return "\n".join([line[1] for line in result])
    return ""
if __name__ == '__main__':

    import os
    absfile=os.path.abspath(__file__)
    dirfile=os.path.dirname(absfile)
    dirfile=os.path.dirname(dirfile)
    file=os.path.join(dirfile,'data/test.png')
    text = get_ocr(file)
    print(text)  # 如果输出了图片中的文字 → OCR 没问题
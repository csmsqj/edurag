#文本预处理
# utils/preprocess.py
import jieba
def preprocess_text(text):
    """
    对输入文本进行预处理：中文分词 + 小写规范化

    参数：
        text (str): 需要分词的原始中文文本，例如 "什么是机器学习"

    返回：
        list[str]: 分词后的词语列表，例如 ["什么", "是", "机器", "学习"]

    为什么用 jieba.lcut 而不是 jieba.cut：
        - jieba.cut() 返回的是生成器（generator），需要 list() 转换才能得到列表
        - jieba.lcut() 直接返回列表（list），使用更方便
        - 在数据量不大的场景下，两者性能差异可忽略

    为什么要转小写 lower()：
        - 统一英文大小写，避免 "Python" 和 "python" 被当作不同词
        - 对中文没有影响（中文没有大小写概念），但对混合了英文的问题有用
    """
    return [word.lower() for word in jieba.lcut(text)]
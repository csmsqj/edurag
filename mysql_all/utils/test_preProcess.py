#文本预处理
# utils/preprocess.py
import jieba
import re

# 匹配所有标点符号、空白字符、特殊符号
_PUNCTUATION_RE = re.compile(r'[^\w]|[\d_]', re.UNICODE)
# 停用词（高频无意义词）
_STOPWORDS = frozenset([
    '的', '了', '是', '在', '我', '有', '和', '就', '不', '人', '都', '一',
    '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着',
    '没有', '看', '好', '自己', '这', '他', '她', '它', '们', '那',
    '什么', '怎么', '如何', '吗', '呢', '吧', '啊', '哦', '嗯',
    '可以', '能', '把', '被', '从', '对', '与', '为', '之',
    '还', '又', '但', '而', '或', '及', '等', '中', '个',
])

def preprocess_text(text):
    """
    对输入文本进行预处理：中文分词 + 小写 + 去标点 + 去停用词
    """
    words = jieba.lcut(text)
    result = []
    for word in words:
        w = word.lower().strip()
        if not w:
            continue
        if _PUNCTUATION_RE.fullmatch(w):
            continue
        if w in _STOPWORDS:
            continue
        result.append(w)
    return result
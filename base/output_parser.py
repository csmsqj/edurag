import re
from langchain_core.output_parsers import StrOutputParser


class ThinkingStripper(StrOutputParser):
    """去除 Google Gemma/Gemini 模型输出中的 thinking 标签"""

    def parse(self, text: str) -> str:
        cleaned = re.sub(r'<(?:thought|thinking|think)>.*?</(?:thought|thinking|think)>', '', text, flags=re.DOTALL)
        return cleaned.strip()

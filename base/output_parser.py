import re
from typing import Iterator, AsyncIterator
from langchain_core.output_parsers import StrOutputParser


class ThinkingStripper(StrOutputParser):
    """去除模型输出中的 thinking 标签，支持流式"""

    def parse(self, text: str) -> str:
        cleaned = re.sub(r'<(?:thought|thinking|think)>.*?</(?:thought|thinking|think)>', '', text, flags=re.DOTALL)
        return cleaned.strip()

    def _extract_text(self, chunk) -> str:
        if hasattr(chunk, 'content'):
            return chunk.content
        return str(chunk)

    def _transform(self, input: Iterator, **kwargs) -> Iterator[str]:
        buffer = ""
        in_thinking = False
        outputting = False

        for chunk in input:
            text = self._extract_text(chunk)
            if not text:
                continue

            if outputting:
                yield text
                continue

            buffer += text

            if not in_thinking and re.search(r'<(?:thought|thinking|think)>', buffer):
                in_thinking = True

            if in_thinking:
                close_match = re.search(r'</(?:thought|thinking|think)>', buffer)
                if close_match:
                    outputting = True
                    remainder = buffer[close_match.end():]
                    if remainder:
                        yield remainder
                    buffer = ""
            else:
                if len(buffer) > 20:
                    outputting = True
                    yield buffer
                    buffer = ""

        if buffer:
            cleaned = self.parse(buffer)
            if cleaned:
                yield cleaned

    async def _atransform(self, input: AsyncIterator, **kwargs) -> AsyncIterator[str]:
        buffer = ""
        in_thinking = False
        outputting = False

        async for chunk in input:
            text = self._extract_text(chunk)
            if not text:
                continue

            if outputting:
                yield text
                continue

            buffer += text

            if not in_thinking and re.search(r'<(?:thought|thinking|think)>', buffer):
                in_thinking = True

            if in_thinking:
                close_match = re.search(r'</(?:thought|thinking|think)>', buffer)
                if close_match:
                    outputting = True
                    remainder = buffer[close_match.end():]
                    if remainder:
                        yield remainder
                    buffer = ""
            else:
                if len(buffer) > 20:
                    outputting = True
                    yield buffer
                    buffer = ""

        if buffer:
            cleaned = self.parse(buffer)
            if cleaned:
                yield cleaned

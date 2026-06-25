# api.py

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import json
import uvicorn
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from new_main import IntegratedQA

app = FastAPI(title="EduRAG 智能问答系统")

# 跨域配置（开发阶段允许所有来源）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件（WebUI）
app.mount("/static", StaticFiles(directory="static"), name="static")

# 全局初始化问答系统（只初始化一次，避免重复连接数据库）
qa_system = IntegratedQA()


class QueryRequest(BaseModel):
    """请求体模型（Pydantic 自动校验）"""
    query: str
    source_filter: Optional[str] = None
    session_id: Optional[str] = None


def generate_stream(query: str, source_filter: str = None, session_id: str = None):
    """
    SSE 流式生成器
    将 IntegratedQASystem.query() 的 yield 转为 SSE 格式
    """
    for token, is_complete in qa_system.answer_question(query, session_id=session_id):
        # 构造 SSE data 行
        data = json.dumps(
            {"token": token, "done": is_complete},
            ensure_ascii=False   # 保留中文，不转 unicode 转义
        )
        yield f"data: {data}\n\n"    # SSE 协议：data: + 内容 + 两个换行


@app.post("/api/chat")
async def chat(request: QueryRequest):
    """
    聊天接口 — 返回 SSE 流
    Content-Type: text/event-stream
    """
    return StreamingResponse(
        generate_stream(request.query, request.source_filter, request.session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",         # 禁止缓存
            "Connection": "keep-alive",          # 保持长连接
            "X-Accel-Buffering": "no"            # Nginx 禁用缓冲（如果有反代）
        }
    )

@app.get("/")
async def root():
    return RedirectResponse(url="/static/index.html")


@app.get("/api/health")
async def health():
    """健康检查"""
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=13280)
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from app import agent, cache, config, llm, rag, tables
from app.db import close_pools, init_schema, open_pools, pool
from app.etl import SUPPORTED, ingest

logger = logging.getLogger("uvicorn.error")
ALLOWED_SUFFIXES = SUPPORTED
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_schema()
    open_pools()
    yield
    close_pools()


app = FastAPI(title="local-rag-kit", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    use_cache: bool = True


class AgentRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    target: Literal["local", "cloud"] = "local"


UPSTREAM_ERRORS = {401: "雲端 API key 無效", 402: "雲端 API 額度不足", 429: "雲端 API 請求過於頻繁"}


def upstream_error(e: httpx.HTTPStatusError) -> str:
    # 只回狀態碼對應的固定訊息，不把上游回應內容（可能含帳號、專案資訊）轉給前端
    logger.warning("upstream %s: %s", e.response.status_code, e.response.text[:500])
    return UPSTREAM_ERRORS.get(e.response.status_code, f"模型服務錯誤（HTTP {e.response.status_code}）")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/health")
def health() -> dict:
    with pool.connection() as conn:
        chunks = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    ollama = "skip"
    if "ollama" in (config.LLM_PROVIDER, config.EMBED_PROVIDER):
        ollama = httpx.get(f"{config.OLLAMA_URL}/api/version", timeout=3).json()["version"]
    return {"postgres": "ok", "chunks": chunks, "redis": cache.r.ping(), "ollama": ollama,
            "llm": f"{config.LLM_PROVIDER}:{config.CHAT_MODEL}", "embed": f"{config.EMBED_PROVIDER}:{config.EMBED_MODEL}"}


async def read_uploads(files: list[UploadFile], allowed: set[str]) -> list[tuple[str, bytes]]:
    out = []
    for f in files:
        # 檔名只用來當 source / 表名顯示，不拿去拼檔案路徑；表名進 SQL 前一律經 sql.Identifier 跳脫
        name = Path(f.filename or "upload").name
        if Path(name).suffix.lower() not in allowed:
            raise HTTPException(400, f"不支援的檔案類型：{name}")
        data = await f.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"檔案超過 20MB：{name}")
        out.append((name, data))
    return out


@app.post("/ingest")
async def ingest_files(files: list[UploadFile]) -> list[dict]:
    return [ingest(name, data) for name, data in await read_uploads(files, ALLOWED_SUFFIXES)]


@app.post("/tables")
async def import_tables(files: list[UploadFile]) -> list[dict]:
    return [r for name, data in await read_uploads(files, tables.TABULAR) for r in tables.import_file(name, data)]


@app.get("/tables")
def list_tables() -> list[dict]:
    return tables.describe()


@app.get("/search")
def search(q: str, k: int = config.TOP_K) -> list[dict]:
    return rag.retrieve(llm.embed([q])[0], min(k, 20))


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    t0 = time.perf_counter()

    def elapsed() -> int:
        return round((time.perf_counter() - t0) * 1000)

    if req.use_cache and (hit := cache.get_exact(req.question)):
        return {**hit, "cache": "exact", "ms": elapsed()}

    embedding = llm.embed([req.question])[0]
    if req.use_cache and config.SEMANTIC_CACHE and (hit := cache.get_semantic(embedding)):
        return {**hit, "cache": "semantic", "ms": elapsed()}

    contexts = rag.retrieve(embedding)
    if not contexts:
        return {"answer": "知識庫目前是空的，請先 /ingest 文件", "sources": [], "cache": "miss", "ms": elapsed()}

    result = {"answer": llm.chat(rag.build_messages(req.question, contexts)), "sources": rag.sources_of(contexts)}
    if req.use_cache:
        save_cache(req.question, embedding, result)
    return {**result, "cache": "miss", "ms": elapsed()}


def save_cache(question: str, embedding: list[float], result: dict) -> None:
    cache.set_exact(question, result)
    if config.SEMANTIC_CACHE:
        cache.set_semantic(question, embedding, result)


def sse(data, event: str | None = None) -> str:
    head = f"event: {event}\n" if event else ""
    return f"{head}data: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/ask/stream")
def ask_stream(req: AskRequest) -> StreamingResponse:
    t0 = time.perf_counter()

    def done(layer: str) -> str:
        return sse({"cache": layer, "ms": round((time.perf_counter() - t0) * 1000)}, "done")

    def events():
        if req.use_cache and (hit := cache.get_exact(req.question)):
            yield sse(hit["sources"], "sources")
            yield sse(hit["answer"])
            yield done("exact")
            return

        embedding = llm.embed([req.question])[0]
        if req.use_cache and config.SEMANTIC_CACHE and (hit := cache.get_semantic(embedding)):
            yield sse(hit["sources"], "sources")
            yield sse(hit["answer"])
            yield done("semantic")
            return

        contexts = rag.retrieve(embedding)
        sources = rag.sources_of(contexts)
        yield sse(sources, "sources")
        if not contexts:
            yield sse("知識庫目前是空的，請先上傳文件")
            yield done("miss")
            return

        parts = []
        for token in llm.chat_stream(rag.build_messages(req.question, contexts)):
            parts.append(token)
            yield sse(token)
        if req.use_cache:
            save_cache(req.question, embedding, {"answer": "".join(parts), "sources": sources})
        yield done("miss")

    return StreamingResponse(events(), media_type="text/event-stream")


@app.post("/agent")
def run_agent(req: AgentRequest) -> dict:
    try:
        events = list(agent.run(req.question, req.target))
    except httpx.HTTPStatusError as e:
        raise HTTPException(502, upstream_error(e))
    answer = next((e["content"] for e in events if e["type"] == "answer"), "")
    return {"answer": answer, "trace": [e for e in events if e["type"].startswith("tool")], **events[-1]}


@app.post("/agent/stream")
def run_agent_stream(req: AgentRequest) -> StreamingResponse:
    def events():
        try:
            for e in agent.run(req.question, req.target):
                yield sse(e, e["type"])
        except httpx.HTTPStatusError as e:
            yield sse({"message": upstream_error(e)}, "error")

    return StreamingResponse(events(), media_type="text/event-stream")


@app.delete("/cache")
def clear_cache() -> dict:
    return {"cleared_exact_keys": cache.clear()}

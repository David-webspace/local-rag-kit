import hashlib
import json

import redis
from psycopg.types.json import Jsonb

from app import config
from app.db import pool

r = redis.Redis.from_url(config.REDIS_URL, decode_responses=True)
PREFIX = "rag:exact:"


def _normalize(question: str) -> str:
    return " ".join(question.lower().split())


def _key(question: str) -> str:
    # key 帶上模型名稱，換生成或 embedding 模型後不會拿到舊設定的答案
    raw = f"{config.CHAT_MODEL}|{config.EMBED_MODEL}|{config.TOP_K}|{_normalize(question)}"
    return PREFIX + hashlib.sha256(raw.encode()).hexdigest()


def get_exact(question: str) -> dict | None:
    hit = r.get(_key(question))
    return json.loads(hit) if hit else None


def set_exact(question: str, result: dict) -> None:
    r.set(_key(question), json.dumps(result, ensure_ascii=False), ex=config.CACHE_TTL)


def get_semantic(embedding: list[float]) -> dict | None:
    with pool.connection() as conn:
        row = conn.execute(
            """
            SELECT question, answer, sources, 1 - (embedding <=> %s::vector) AS similarity
            FROM semantic_cache
            WHERE created_at > now() - make_interval(secs => %s)
            ORDER BY embedding <=> %s::vector
            LIMIT 1
            """,
            (embedding, config.CACHE_TTL, embedding),
        ).fetchone()
    if row and row[3] >= config.SEMANTIC_CACHE_THRESHOLD:
        return {"answer": row[1], "sources": row[2], "matched_question": row[0], "similarity": round(row[3], 4)}
    return None


def set_semantic(question: str, embedding: list[float], result: dict) -> None:
    with pool.connection() as conn:
        conn.execute(
            "INSERT INTO semantic_cache (question, answer, sources, embedding) VALUES (%s, %s, %s, %s)",
            (question, result["answer"], Jsonb(result["sources"]), embedding),
        )


def clear() -> int:
    keys = list(r.scan_iter(f"{PREFIX}*"))
    if keys:
        r.delete(*keys)
    with pool.connection() as conn:
        conn.execute("TRUNCATE semantic_cache")
    return len(keys)

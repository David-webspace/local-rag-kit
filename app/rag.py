from app import config
from app.db import pool

# 小模型容易把「拒答句」當成預設答案，所以先要求作答，拒答條件放最後並限定「完全沒提到」
SYSTEM_PROMPT = """你是知識庫問答助理。請仔細閱讀「參考資料」，用其中的內容回答問題。
- 先直接給出結論，再補充相關條件或限制。
- 句尾用 [編號] 標註依據的參考資料。
- 使用繁體中文，簡潔回答。
- 只有當所有參考資料都完全沒有提到問題的主題時，才回答「資料中沒有相關資訊」；不要自行推測資料以外的內容。"""


def retrieve(embedding: list[float], k: int = config.TOP_K) -> list[dict]:
    with pool.connection() as conn:
        rows = conn.execute(
            """
            SELECT d.source, c.chunk_index, c.content, 1 - (c.embedding <=> %s::vector) AS score
            FROM chunks c JOIN documents d ON d.id = c.document_id
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s
            """,
            (embedding, embedding, k),
        ).fetchall()
    return [{"source": s, "chunk_index": i, "content": c, "score": round(sc, 4)} for s, i, c, sc in rows]


def build_messages(question: str, contexts: list[dict]) -> list[dict]:
    refs = "\n\n".join(f"[{n}] ({c['source']})\n{c['content']}" for n, c in enumerate(contexts, 1))
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"參考資料：\n{refs}\n\n問題：{question}"},
    ]


def sources_of(contexts: list[dict]) -> list[dict]:
    return [{**{k: c[k] for k in ("source", "chunk_index", "score")}, "preview": c["content"][:160]} for c in contexts]

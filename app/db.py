import logging

import psycopg
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from app import config

logger = logging.getLogger("uvicorn.error")


def _drop_if_dim_changed(conn: psycopg.Connection) -> None:
    # pgvector 欄位的維度存在 atttypmod；跟 EMBED_DIM 不同代表換了 embedding 模型，舊向量全部不可用
    row = conn.execute(
        "SELECT atttypmod FROM pg_attribute WHERE attrelid = to_regclass('chunks') AND attname = 'embedding'"
    ).fetchone()
    if row and row[0] != config.EMBED_DIM:
        logger.warning("EMBED_DIM %s -> %s：重建向量表，請重新執行 ETL", row[0], config.EMBED_DIM)
        conn.execute("DROP TABLE IF EXISTS chunks, documents, semantic_cache")


HNSW_MAX_DIM = 2000


def init_schema() -> None:
    if config.EMBED_DIM > HNSW_MAX_DIM:
        raise ValueError(
            f"EMBED_DIM={config.EMBED_DIM} 超過 pgvector HNSW 上限 {HNSW_MAX_DIM}；"
            "請改用 1536 或 768（text-embedding-3-* / gemini-embedding-* 會自動帶 dimensions 參數）"
        )
    with psycopg.connect(config.PG_DSN, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        _drop_if_dim_changed(conn)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id          BIGSERIAL PRIMARY KEY,
                source      TEXT UNIQUE NOT NULL,
                sha256      TEXT NOT NULL,
                ingested_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS chunks (
                id          BIGSERIAL PRIMARY KEY,
                document_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                chunk_index INT NOT NULL,
                content     TEXT NOT NULL,
                embedding   vector({config.EMBED_DIM}) NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw "
            "ON chunks USING hnsw (embedding vector_cosine_ops)"
        )
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS semantic_cache (
                id         BIGSERIAL PRIMARY KEY,
                question   TEXT NOT NULL,
                answer     TEXT NOT NULL,
                sources    JSONB NOT NULL,
                embedding  vector({config.EMBED_DIM}) NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS semantic_cache_embedding_hnsw "
            "ON semantic_cache USING hnsw (embedding vector_cosine_ops)"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS products (
                sku         TEXT PRIMARY KEY,
                name        TEXT NOT NULL,
                category    TEXT NOT NULL,
                price       INT NOT NULL,
                stock       INT NOT NULL,
                description TEXT NOT NULL
            )
            """
        )


pool = ConnectionPool(config.PG_DSN, min_size=1, max_size=5, open=False, configure=register_vector)

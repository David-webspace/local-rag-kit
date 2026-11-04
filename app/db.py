import psycopg
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from app import config


def init_schema() -> None:
    with psycopg.connect(config.PG_DSN, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
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


pool = ConnectionPool(config.PG_DSN, min_size=1, max_size=5, open=False, configure=register_vector)

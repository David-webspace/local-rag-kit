import csv
import hashlib
import io
import re
import sys
from pathlib import Path

from pypdf import PdfReader

from app import cache, config, llm
from app.db import init_schema, pool

EMBED_BATCH = 32


def extract(filename: str, data: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        return "\n\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(data)).pages)
    text = data.decode("utf-8-sig")
    if suffix == ".csv":
        # 每一列攤平成「欄位: 值」，ingest 時一列就是一個 chunk
        rows = csv.DictReader(io.StringIO(text))
        return "\n\n".join("；".join(f"{k}: {v}" for k, v in row.items() if v) for row in rows)
    return text


def clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("　", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk(text: str, size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP) -> list[str]:
    pieces: list[str] = []
    for para in (p.strip() for p in text.split("\n\n")):
        if not para:
            continue
        while len(para) > size:
            pieces.append(para[:size])
            para = para[size - overlap :]
        pieces.append(para)

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        # Markdown 標題是主題邊界：直接斷開，不帶 overlap
        if current and piece.startswith("#"):
            chunks.append(current)
            current = piece
        elif current and len(current) + len(piece) + 2 > size:
            chunks.append(current)
            current = current[-overlap:] + "\n\n" + piece
        else:
            current = f"{current}\n\n{piece}" if current else piece
    if current:
        chunks.append(current)
    return chunks


def ingest(filename: str, data: bytes) -> dict:
    sha = hashlib.sha256(data).hexdigest()
    with pool.connection() as conn:
        row = conn.execute("SELECT sha256 FROM documents WHERE source = %s", (filename,)).fetchone()
    if row and row[0] == sha:
        return {"source": filename, "status": "unchanged", "chunks": 0}

    text = clean(extract(filename, data))
    chunks = text.split("\n\n") if filename.lower().endswith(".csv") else chunk(text)
    vectors: list[list[float]] = []
    for i in range(0, len(chunks), EMBED_BATCH):
        vectors.extend(llm.embed(chunks[i : i + EMBED_BATCH]))

    # 舊版文件整份替換（ON DELETE CASCADE 會一起清掉舊 chunks），全部包在同一個 transaction
    with pool.connection() as conn, conn.transaction():
        conn.execute("DELETE FROM documents WHERE source = %s", (filename,))
        doc_id = conn.execute(
            "INSERT INTO documents (source, sha256) VALUES (%s, %s) RETURNING id", (filename, sha)
        ).fetchone()[0]
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO chunks (document_id, chunk_index, content, embedding) VALUES (%s, %s, %s, %s)",
                [(doc_id, i, c, v) for i, (c, v) in enumerate(zip(chunks, vectors))],
            )

    # 知識庫變了，舊答案可能過期
    cache.clear()
    return {"source": filename, "status": "updated" if row else "created", "chunks": len(chunks)}


if __name__ == "__main__":
    init_schema()
    pool.open()
    for arg in sys.argv[1:]:
        for path in sorted(Path(arg).rglob("*")) if Path(arg).is_dir() else [Path(arg)]:
            if path.is_file() and path.suffix.lower() in {".pdf", ".md", ".txt", ".csv"}:
                print(ingest(path.name, path.read_bytes()))
    pool.close()

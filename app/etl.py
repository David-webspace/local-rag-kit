import csv
import hashlib
import io
import json
import re
import sys
from pathlib import Path

from docx import Document
from openpyxl import load_workbook
from pypdf import PdfReader

from app import cache, config, llm
from app.db import init_schema, pool

EMBED_BATCH = 32
SUPPORTED = {".pdf", ".docx", ".md", ".txt", ".csv", ".xlsx", ".json"}


def _record(row: dict, prefix: str = "") -> str:
    fields = "；".join(
        f"{k}: {json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v}"
        for k, v in row.items()
        if v not in (None, "")
    )
    return f"{prefix}{fields}"


def _xlsx_records(data: bytes) -> list[str]:
    records = []
    wb = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    for ws in wb.worksheets:
        rows = [r for r in ws.iter_rows(values_only=True) if any(c not in (None, "") for c in r)]
        if not rows:
            continue
        header = [str(h) if h is not None else f"欄{i + 1}" for i, h in enumerate(rows[0])]
        records += [_record(dict(zip(header, r)), f"[{ws.title}] ") for r in rows[1:]]
    return records


def _docx_text(data: bytes) -> str:
    blocks = []
    for item in Document(io.BytesIO(data)).iter_inner_content():
        if hasattr(item, "rows"):
            rows = [[c.text.strip() for c in r.cells] for r in item.rows]
            blocks += [_record(dict(zip(rows[0], r))) for r in rows[1:]]
            continue
        text = item.text.strip()
        if not text:
            continue
        # Word 標題樣式轉成 Markdown 標題，chunk() 才能依標題斷段
        style = item.style.name if item.style is not None else ""
        level = 1 if style == "Title" else int(style[8:]) if style.startswith("Heading ") and style[8:].isdigit() else 0
        blocks.append(f"{'#' * level} {text}" if level else text)
    return "\n\n".join(blocks)


def document_text(filename: str, data: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        return clean("\n\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(data)).pages))
    if suffix == ".docx":
        return clean(_docx_text(data))
    return clean(data.decode("utf-8-sig"))


def split(filename: str, data: bytes) -> list[str]:
    """表格類資料一筆一個 chunk（同一筆資料不會被切開）；文件類資料走 chunk()"""
    suffix = Path(filename).suffix.lower()
    if suffix == ".xlsx":
        return [clean(r) for r in _xlsx_records(data)]
    if suffix in (".pdf", ".docx"):
        return chunk(document_text(filename, data))

    text = data.decode("utf-8-sig")
    if suffix == ".csv":
        return [clean(_record(row)) for row in csv.DictReader(io.StringIO(text))]
    if suffix == ".json":
        obj = json.loads(text)
        if isinstance(obj, list) and all(isinstance(o, dict) for o in obj):
            return [clean(_record(o)) for o in obj]
        return chunk(clean(json.dumps(obj, ensure_ascii=False, indent=2)))
    return chunk(clean(text))


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
        # Markdown 標題是主題邊界：直接斷開，不帶 overlap；但只有標題的 chunk 沒有內容，併進下一段
        only_headings = all(p.startswith("#") and "\n" not in p for p in current.split("\n\n"))
        if current and piece.startswith("#") and not only_headings:
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

    chunks = [c for c in split(filename, data) if c]
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
            if path.is_file() and path.suffix.lower() in SUPPORTED:
                print(ingest(path.name, path.read_bytes()))
    pool.close()

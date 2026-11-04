"""LLM 結構化擷取：一批非結構化文字 + JSON Schema → 驗證過的資料表

uv run --env-file .env python -m app.extract --schema examples/extract/complaint.schema.json \
    --table complaints examples/extract/complaints
"""

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from jsonschema import Draft202012Validator
from psycopg import sql
from psycopg.types.json import Jsonb

from app import config, llm, tables
from app.db import close_pools, init_schema, open_pools, pool
from app.etl import SUPPORTED, document_text

MAX_RETRIES = 2
MAX_DOC_CHARS = 12000

SYSTEM_PROMPT = """你是資料擷取器。依照 JSON Schema 從文件中抽取欄位，只輸出一個 JSON 物件，不要任何說明文字。
- 只能使用文件中明確寫出的資訊；文件沒有的欄位依 schema 填 null，不要推測。
- 嚴格遵守每個欄位的型別、enum、格式與 description。"""


def _main_type(prop: dict) -> str:
    t = prop.get("type", "string")
    if isinstance(t, list):
        t = next((x for x in t if x != "null"), "string")
    return t


def pg_type(prop: dict) -> str:
    t = _main_type(prop)
    if t == "string":
        return {"date": "date", "date-time": "timestamptz"}.get(prop.get("format"), "text")
    return {"integer": "bigint", "number": "numeric", "boolean": "boolean"}.get(t, "jsonb")


def _parse(text: str) -> dict:
    text = text.strip()
    # 有些模型即使要求 JSON 也會包在 ```json 區塊裡
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    return json.loads(fenced.group(1) if fenced else text)


def extract_one(text: str, schema: dict, validator: Draft202012Validator, provider: str, model: str) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"JSON Schema：\n{json.dumps(schema, ensure_ascii=False)}\n\n文件：\n{text[:MAX_DOC_CHARS]}"},
    ]
    usage = {"input": 0, "output": 0}
    error = ""
    for attempt in range(MAX_RETRIES + 1):
        raw, u = llm.chat_json(messages, schema, provider, model)
        usage = {k: usage[k] + u[k] for k in usage}
        try:
            record = _parse(raw)
            problems = [f"{'/'.join(map(str, e.absolute_path)) or '(root)'}：{e.message}" for e in validator.iter_errors(record)]
        except json.JSONDecodeError as e:
            problems = [f"不是合法 JSON：{e}"]
        if not problems:
            return {"record": record, "attempts": attempt + 1, "usage": usage}
        error = "；".join(problems)
        # 驗證錯誤原樣回給模型，讓它針對問題修正，而不是盲目重問
        messages += [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": f"輸出不符合 schema：{error}\n請修正後只輸出 JSON。"},
        ]
    return {"error": error, "attempts": MAX_RETRIES + 1, "usage": usage}


def load_documents(paths: list[Path], text_column: str | None) -> list[tuple[str, str]]:
    """回傳 [(來源, 文字)]。表格檔搭配 --text-column 時一列一筆，其他檔案一個檔一筆"""
    docs = []
    for p in paths:
        files = sorted(f for f in p.rglob("*") if f.is_file()) if p.is_dir() else [p]
        for f in files:
            suffix = f.suffix.lower()
            if text_column and suffix in tables.TABULAR:
                for sheet, rows in tables.read_sheets(f.name, f.read_bytes()).items():
                    header = [str(h).strip() if h is not None else "" for h in rows[0]]
                    idx = header.index(text_column)
                    docs += [(f"{sheet}#{i}", str(r[idx])) for i, r in enumerate(rows[1:], 1) if r[idx]]
            elif suffix in SUPPORTED | {".eml"}:
                docs.append((f.name, document_text(f.name, f.read_bytes())))
    return docs


def save(table: str, schema: dict, records: list[tuple[str, dict]]) -> None:
    props = schema["properties"]
    t = sql.Identifier(config.DATA_SCHEMA, tables.ident(table))
    cols = [sql.SQL("{} {}").format(sql.Identifier(tables.ident(k)), sql.SQL(pg_type(v))) for k, v in props.items()]
    cols.append(sql.SQL("_source text"))
    names = sql.SQL(", ").join([sql.Identifier(tables.ident(k)) for k in props] + [sql.Identifier("_source")])
    marks = sql.SQL(", ").join(sql.Placeholder() * (len(props) + 1))
    rows = [
        [Jsonb(rec.get(k)) if pg_type(v) == "jsonb" else rec.get(k) for k, v in props.items()] + [src]
        for src, rec in records
    ]
    with pool.connection() as conn, conn.transaction():
        conn.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(t))
        conn.execute(sql.SQL("CREATE TABLE {} ({})").format(t, sql.SQL(", ").join(cols)))
        with conn.cursor() as cur:
            cur.executemany(sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(t, names, marks), rows)


def run(docs: list[tuple[str, str]], schema: dict, table: str, target: str, workers: int) -> dict:
    provider, model = config.AGENT_TARGETS[target]
    validator = Draft202012Validator(schema, format_checker=Draft202012Validator.FORMAT_CHECKER)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(lambda d: extract_one(d[1], schema, validator, provider, model), docs))

    ok = [(src, r["record"]) for (src, _), r in zip(docs, results) if "record" in r]
    failed = [{"source": src, "error": r["error"]} for (src, _), r in zip(docs, results) if "error" in r]
    save(table, schema, ok)
    return {
        "table": tables.ident(table),
        "model": model,
        "total": len(docs),
        "succeeded": len(ok),
        "failed": failed,
        "retried": sum(1 for r in results if r["attempts"] > 1),
        "usage": {k: sum(r["usage"][k] for r in results) for k in ("input", "output")},
        "ms": round((time.perf_counter() - t0) * 1000),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="LLM 結構化擷取 → data schema 資料表")
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--schema", required=True, type=Path)
    ap.add_argument("--table", required=True)
    ap.add_argument("--target", choices=["local", "cloud"], default="local")
    ap.add_argument("--text-column", help="CSV / Excel 中要擷取的文字欄位")
    ap.add_argument("--workers", type=int, default=1, help="並行數；雲端可設 4–8，地端建議 1")
    args = ap.parse_args()

    init_schema()
    open_pools()
    summary = run(load_documents(args.paths, args.text_column), json.loads(args.schema.read_text()),
                  args.table, args.target, args.workers)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    close_pools()

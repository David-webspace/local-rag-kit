"""任意 CSV / Excel → data schema 資料表；以及用唯讀角色執行 LLM 產生的 SQL"""

import csv
import io
import re
import sys
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

from openpyxl import load_workbook
from psycopg import sql

from app import config
from app.db import close_pools, init_schema, open_pools, pool, reader_pool

TABULAR = {".csv", ".xlsx"}
MAX_RESULT_ROWS = 50
MAX_SQL_CHARS = 4000
PG_IDENT_BYTES = 63

INT_RE = re.compile(r"^-?\d{1,3}(,\d{3})+$|^-?\d+$")
NUM_RE = re.compile(r"^-?(\d{1,3}(,\d{3})+|\d+)?\.\d+$")
DATE_RE = re.compile(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}$")
TS_RE = re.compile(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}[ T]\d{1,2}:\d{2}(:\d{2})?$")


def ident(name: str) -> str:
    # 英文一律轉小寫：模型常寫出沒加引號的 SKU，PostgreSQL 會折成 sku，原名是大寫就對不到
    name = re.sub(r"\s+", "_", str(name).strip()).lower() or "col"
    while len(name.encode()) > PG_IDENT_BYTES:
        name = name[:-1]
    return name


# ── 讀檔 ─────────────────────────────────────────────────────

def read_sheets(filename: str, data: bytes) -> dict[str, list[list]]:
    """回傳 {表名: [表頭, 列, 列…]}；Excel 每個工作表一張表"""
    stem = Path(filename).stem
    if Path(filename).suffix.lower() == ".csv":
        return {stem: list(csv.reader(io.StringIO(data.decode("utf-8-sig"))))}
    wb = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    sheets = {ws.title: [list(r) for r in ws.iter_rows(values_only=True)] for ws in wb.worksheets}
    sheets = {k: v for k, v in sheets.items() if v}
    if len(sheets) == 1:
        return {stem: next(iter(sheets.values()))}
    return {f"{stem}_{title}": rows for title, rows in sheets.items()}


def _empty(v) -> bool:
    return v is None or (isinstance(v, str) and not v.strip())


# ── 型別推斷 ──────────────────────────────────────────────────

def infer_type(values: list) -> str:
    """選一個讓所有非空值都能轉換成功的最窄型別，否則 text；匯入時不會有轉型失敗的列"""
    vals = [v for v in values if not _empty(v)]
    if not vals:
        return "text"
    if all(isinstance(v, bool) for v in vals):
        return "boolean"
    strs = [str(v).strip() for v in vals if not isinstance(v, bool)]
    if len(strs) != len(vals):
        return "text"
    if all(isinstance(v, int) or INT_RE.match(s) for v, s in zip(vals, strs)):
        return "bigint"
    if all(isinstance(v, (int, float)) or INT_RE.match(s) or NUM_RE.match(s) for v, s in zip(vals, strs)):
        return "numeric"
    if all(isinstance(v, datetime) and v.time() == datetime.min.time() or DATE_RE.match(s) for v, s in zip(vals, strs)):
        return "date"
    if all(isinstance(v, datetime) or TS_RE.match(s) or DATE_RE.match(s) for v, s in zip(vals, strs)):
        return "timestamp"
    return "text"


def convert(v, pg_type: str):
    if _empty(v):
        return None
    if pg_type in ("bigint", "numeric") and isinstance(v, str):
        s = v.strip().replace(",", "")
        return int(s) if pg_type == "bigint" else Decimal(s)
    if pg_type == "date" and isinstance(v, str):
        return date(*map(int, re.split(r"[-/]", v.strip())))
    if pg_type == "date" and isinstance(v, datetime):
        return v.date()
    if pg_type == "timestamp" and isinstance(v, str):
        return datetime.fromisoformat(v.strip().replace("/", "-").replace(" ", "T"))
    if pg_type == "text":
        return str(v).strip()
    return v


# ── 匯入 ─────────────────────────────────────────────────────

def _columns(header: list) -> list[str]:
    cols, seen = [], {}
    for i, h in enumerate(header):
        name = ident(h) if not _empty(h) else f"欄{i + 1}"
        seen[name] = seen.get(name, 0) + 1
        cols.append(name if seen[name] == 1 else f"{name}_{seen[name]}")
    return cols


def import_rows(table: str, rows: list[list]) -> dict:
    rows = [r for r in rows if any(not _empty(v) for v in r)]
    if len(rows) < 2:
        return {"table": table, "rows": 0, "error": "沒有資料列"}
    cols = _columns(rows[0])
    width = len(cols)
    body = [(list(r) + [None] * width)[:width] for r in rows[1:]]
    types = [infer_type([r[i] for r in body]) for i in range(width)]
    values = [tuple(convert(r[i], types[i]) for i in range(width)) for r in body]

    t = sql.Identifier(config.DATA_SCHEMA, ident(table))
    col_defs = sql.SQL(", ").join(sql.SQL("{} {}").format(sql.Identifier(c), sql.SQL(ty)) for c, ty in zip(cols, types))
    with pool.connection() as conn, conn.transaction():
        conn.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(t))
        conn.execute(sql.SQL("CREATE TABLE {} ({})").format(t, col_defs))
        with conn.cursor().copy(sql.SQL("COPY {} FROM STDIN").format(t)) as cp:
            for v in values:
                cp.write_row(v)

    return {
        "table": ident(table),
        "rows": len(values),
        "duplicate_rows": len(values) - len(set(values)),
        "columns": [
            {"name": c, "type": ty, "nulls": sum(v[i] is None for v in values)}
            for i, (c, ty) in enumerate(zip(cols, types))
        ],
    }


def import_file(filename: str, data: bytes) -> list[dict]:
    return [import_rows(name, rows) for name, rows in read_sheets(filename, data).items()]


# ── 給 LLM 的 schema 描述與唯讀查詢 ───────────────────────────────

def describe(sample_rows: int = 2) -> list[dict]:
    with pool.connection() as conn:
        cols = conn.execute(
            """
            SELECT table_name, column_name, data_type FROM information_schema.columns
            WHERE table_schema = %s ORDER BY table_name, ordinal_position
            """,
            (config.DATA_SCHEMA,),
        ).fetchall()
        tables: dict[str, dict] = {}
        for t, c, ty in cols:
            tables.setdefault(t, {"table": t, "columns": []})["columns"].append({"name": c, "type": ty})
        for t, info in tables.items():
            ref = sql.Identifier(config.DATA_SCHEMA, t)
            info["rows"] = conn.execute(sql.SQL("SELECT count(*) FROM {}").format(ref)).fetchone()[0]
            cur = conn.execute(sql.SQL("SELECT * FROM {} LIMIT %s").format(ref), (sample_rows,))
            info["sample"] = [dict(zip([d.name for d in cur.description], r)) for r in cur.fetchall()]
    return list(tables.values())


def run_sql(query: str) -> dict:
    query = query.strip().rstrip(";")
    if len(query) > MAX_SQL_CHARS:
        return {"error": f"SQL 超過 {MAX_SQL_CHARS} 字元"}
    if not re.match(r"^(select|with)\b", query, re.IGNORECASE):
        return {"error": "只允許 SELECT 或 WITH 查詢"}
    # 真正的防線是唯讀角色（只有 data schema 的 SELECT 權限、唯讀 transaction、5 秒 timeout），上面只是提早給模型清楚的錯誤
    with reader_pool.connection() as conn:
        try:
            cur = conn.execute(query)
            rows = cur.fetchmany(MAX_RESULT_ROWS + 1)
        except Exception as e:  # noqa: BLE001 — 錯誤訊息要回給模型讓它修正 SQL
            conn.rollback()
            return {"error": str(e).splitlines()[0]}
        conn.rollback()
    return {
        "columns": [d.name for d in cur.description],
        "rows": [list(r) for r in rows[:MAX_RESULT_ROWS]],
        "truncated": len(rows) > MAX_RESULT_ROWS,
    }


if __name__ == "__main__":
    init_schema()
    open_pools()
    for arg in sys.argv[1:]:
        for path in sorted(Path(arg).rglob("*")) if Path(arg).is_dir() else [Path(arg)]:
            if path.is_file() and path.suffix.lower() in TABULAR:
                for r in import_file(path.name, path.read_bytes()):
                    print(r)
    close_pools()

import csv
import sys
from pathlib import Path

from app.db import init_schema, pool

MAX_ROWS = 20


def load_csv(path: Path) -> int:
    with path.open(encoding="utf-8-sig") as f:
        rows = [
            (r["sku"], r["name"], r["category"], int(r["price"]), int(r["stock"]), r["description"])
            for r in csv.DictReader(f)
        ]
    with pool.connection() as conn, conn.transaction():
        conn.execute("TRUNCATE products")
        with conn.cursor() as cur:
            cur.executemany("INSERT INTO products VALUES (%s, %s, %s, %s, %s, %s)", rows)
    return len(rows)


def categories() -> list[str]:
    with pool.connection() as conn:
        return [r[0] for r in conn.execute("SELECT DISTINCT category FROM products ORDER BY 1")]


def query(
    sku: str | None = None,
    category: str | None = None,
    keyword: str | None = None,
    max_price: int | None = None,
    min_stock: int | None = None,
    max_stock: int | None = None,
    limit: int = 10,
) -> list[dict]:
    # 只開放結構化條件、一律參數化：LLM 產生的值永遠不會被拼進 SQL 字串
    where, params = [], []
    if sku:
        where.append("sku = %s")
        params.append(sku)
    if category:
        where.append("category = %s")
        params.append(category)
    # 模型給的關鍵字詞序不一定和商品名稱相同（「Pro 雲主機」vs「星宇雲主機 Pro」），拆詞後逐一比對
    for word in (keyword or "").split():
        where.append("(name ILIKE %s OR description ILIKE %s)")
        params += [f"%{word}%"] * 2
    if max_price is not None:
        where.append("price <= %s")
        params.append(max_price)
    if min_stock is not None:
        where.append("stock >= %s")
        params.append(min_stock)
    if max_stock is not None:
        where.append("stock <= %s")
        params.append(max_stock)

    sql = "SELECT sku, name, category, price, stock, description FROM products"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY sku LIMIT %s"
    params.append(max(1, min(limit, MAX_ROWS)))

    with pool.connection() as conn:
        cur = conn.execute(sql, params)
        cols = [c.name for c in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


if __name__ == "__main__":
    init_schema()
    pool.open()
    print(f"loaded {load_csv(Path(sys.argv[1]))} products")
    pool.close()

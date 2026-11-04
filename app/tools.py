import json

from app import llm, products, rag, tables

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "語意搜尋公司知識庫（員工手冊、服務條款：請假、報帳、遠端、退款、SLA、計費、技術支援等規定）。"
            "回傳最相關的段落與來源。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "要搜尋的問題或關鍵概念，用完整中文句子效果較好"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_products",
            "description": "查詢商品目錄，取得精確的價格（未稅月費，新台幣）、庫存、規格。"
            "這裡只有商品本身，沒有客戶、訂單、銷售紀錄（那些在 run_sql 的資料表）。所有條件皆可省略，多個條件為 AND。",
            "parameters": {
                "type": "object",
                "properties": {
                    "sku": {"type": "string", "description": "商品代號，例如 SV-200"},
                    "category": {"type": "string", "description": "商品類別，必須完全符合下列之一", "enum": []},
                    "keyword": {"type": "string", "description": "在商品名稱與說明中模糊比對的關鍵字，空白分隔的每個詞都必須出現"},
                    "max_price": {"type": "integer", "description": "月費上限（含）"},
                    "min_stock": {"type": "integer", "description": "庫存下限（含）"},
                    "max_stock": {"type": "integer", "description": "庫存上限（含），查缺貨用 0"},
                    "limit": {"type": "integer", "description": f"最多回傳幾筆，上限 {products.MAX_ROWS}"},
                },
            },
        },
    },
]


def _schema_text(tables_info: list[dict]) -> str:
    lines = []
    for t in tables_info:
        cols = ", ".join(f'"{c["name"]}" {c["type"]}' for c in t["columns"])
        sample = json.dumps(t["sample"], ensure_ascii=False, default=str)
        lines.append(f'- "{t["table"]}"（{t["rows"]} 列）：{cols}\n  範例：{sample}')
    return "\n".join(lines)


def _sql_tool(tables_info: list[dict]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": "run_sql",
            "description": "對匯入的資料表執行唯讀 PostgreSQL 查詢，適合篩選、加總、平均、排名、分組統計、跨表 JOIN。"
            f"只能 SELECT / WITH，最多回傳 {tables.MAX_RESULT_ROWS} 列。"
            "加總、平均、計數一定要在 SQL 裡用 SUM / AVG / COUNT 搭配 GROUP BY 實體欄位完成，"
            "不要撈明細回來自己加（例：SELECT \"客戶\", SUM(\"金額\") FROM \"sales\" GROUP BY \"客戶\"）。"
            "表名與欄名一律加雙引號（欄名可能是中文），跨表時注意兩邊欄名可能不同。可用的表：\n" + _schema_text(tables_info),
            "parameters": {
                "type": "object",
                "properties": {"sql": {"type": "string", "description": "單一 PostgreSQL SELECT 查詢"}},
                "required": ["sql"],
            },
        },
    }


def definitions() -> list[dict]:
    # enum 與資料表 schema 每次從 DB 取，資料變動時不用改程式
    TOOLS[1]["function"]["parameters"]["properties"]["category"]["enum"] = products.categories()
    tables_info = tables.describe()
    return TOOLS + [_sql_tool(tables_info)] if tables_info else TOOLS


def _search_knowledge(query: str) -> list[dict]:
    return [
        {"ref": f"{c['source']}#{c['chunk_index']}", "score": c["score"], "content": c["content"]}
        for c in rag.retrieve(llm.embed([query])[0])
    ]


HANDLERS = {
    "search_knowledge": _search_knowledge,
    "query_products": products.query,
    "run_sql": lambda sql: tables.run_sql(sql),
}


def execute(name: str, arguments: dict) -> str:
    # 錯誤回給模型而不是丟例外：讓它看到錯誤訊息後自行修正參數重試
    if name not in HANDLERS:
        return json.dumps({"error": f"未知的工具：{name}"}, ensure_ascii=False)
    try:
        result = HANDLERS[name](**arguments)
    except TypeError as e:
        return json.dumps({"error": f"參數錯誤：{e}"}, ensure_ascii=False)
    return json.dumps(result, ensure_ascii=False, default=str)

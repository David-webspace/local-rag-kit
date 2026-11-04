import json

from app import llm, products, rag

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
            "description": "查詢商品資料庫，取得精確的價格（未稅月費，新台幣）、庫存、規格。所有條件皆可省略，多個條件為 AND。",
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


def definitions() -> list[dict]:
    # enum 每次從 DB 取，類別變動時不用改程式
    TOOLS[1]["function"]["parameters"]["properties"]["category"]["enum"] = products.categories()
    return TOOLS


def _search_knowledge(query: str) -> list[dict]:
    return [
        {"ref": f"{c['source']}#{c['chunk_index']}", "score": c["score"], "content": c["content"]}
        for c in rag.retrieve(llm.embed([query])[0])
    ]


HANDLERS = {"search_knowledge": _search_knowledge, "query_products": products.query}


def execute(name: str, arguments: dict) -> str:
    # 錯誤回給模型而不是丟例外：讓它看到錯誤訊息後自行修正參數重試
    if name not in HANDLERS:
        return json.dumps({"error": f"未知的工具：{name}"}, ensure_ascii=False)
    try:
        result = HANDLERS[name](**arguments)
    except TypeError as e:
        return json.dumps({"error": f"參數錯誤：{e}"}, ensure_ascii=False)
    return json.dumps(result, ensure_ascii=False)

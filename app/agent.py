import json
import time
from collections.abc import Iterator
from datetime import date

from app import config, llm, tools

# 今天日期要明講：模型不知道現在是哪一年，「9 月」會自己猜成訓練資料裡的年份
SYSTEM_PROMPT = """你是「星宇雲端科技」的內部助理，可以使用工具查詢資料後回答。今天是 {today}。
- 公司規定、政策、流程類問題 → 用 search_knowledge。
- 商品價格、庫存、規格類問題 → 用 query_products；數字以資料庫為準。
- 匯入資料表的統計、排名、加總、趨勢類問題 → 用 run_sql（若有提供），在 SQL 內完成計算，不要自己心算。
  問題沒說年份時，以資料實際涵蓋的期間為準（可先查 min/max 日期）。
- 問題牽涉多種資料時，相關工具都要用，可以分多次呼叫；SQL 出錯時依錯誤訊息修正後重試。
- 回答只能根據工具「成功」回傳的結果。工具回傳 error 的部分一律視為不知道，絕對不可以推測或補上數字、名稱。
- 回答用繁體中文，句尾標註依據，例如 (handbook.md#2)、(SV-200) 或 (sales 表)。"""

PREVIEW_CHARS = 600
REPEATED_FAILURE = json.dumps(
    {"error": "這個呼叫與先前失敗的呼叫完全相同，不會再執行。請依錯誤訊息修改參數，或直接說明查不到。"},
    ensure_ascii=False,
)


def run(question: str, target: str) -> Iterator[dict]:
    provider, model = config.AGENT_TARGETS[target]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(today=date.today().isoformat())},
        {"role": "user", "content": question},
    ]
    defs = tools.definitions()
    failed: set[str] = set()
    usage = {"input": 0, "output": 0}
    t0 = time.perf_counter()

    for step in range(1, config.AGENT_MAX_STEPS + 1):
        # 最後一步不給工具，強迫模型收斂成答案，避免無限呼叫
        last = step == config.AGENT_MAX_STEPS
        res = llm.chat_tools(messages, None if last else defs, provider, model)
        usage["input"] += res["usage"]["input"]
        usage["output"] += res["usage"]["output"]
        messages.append(res["message"])

        if not res["tool_calls"]:
            yield {"type": "answer", "content": res["content"]}
            break

        for call in res["tool_calls"]:
            yield {"type": "tool_call", "step": step, "name": call["name"], "arguments": call["arguments"]}
            t1 = time.perf_counter()
            key = f"{call['name']}:{json.dumps(call['arguments'], sort_keys=True, ensure_ascii=False)}"
            if call["arguments"] is None:
                output = json.dumps({"error": "arguments 不是合法 JSON"}, ensure_ascii=False)
            elif key in failed:
                # 小模型常把同一個失敗呼叫原封不動重送，直到步數用完
                output = REPEATED_FAILURE
            else:
                output = tools.execute(call["name"], call["arguments"])
                if output.startswith('{"error"'):
                    failed.add(key)
            yield {
                "type": "tool_result", "step": step, "name": call["name"],
                "preview": output[:PREVIEW_CHARS], "ms": round((time.perf_counter() - t1) * 1000),
            }
            messages.append(llm.tool_message(provider, call, output))

    yield {
        "type": "done", "target": target, "model": model, "steps": step, "usage": usage,
        "ms": round((time.perf_counter() - t0) * 1000),
    }

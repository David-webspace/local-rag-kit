import json
import time
from collections.abc import Iterator

from app import config, llm, tools

SYSTEM_PROMPT = """你是「星宇雲端科技」的內部助理，可以使用工具查詢資料後回答。
- 公司規定、政策、流程類問題 → 用 search_knowledge。
- 商品價格、庫存、規格類問題 → 用 query_products；數字以資料庫為準。
- 問題同時牽涉兩者時，兩個工具都要用，可以分多次呼叫。
- 回答只能根據工具結果，不要編造；查不到就直接說查不到。
- 回答用繁體中文，句尾標註依據，例如 (handbook.md#3) 或 (SV-200)。"""

PREVIEW_CHARS = 600


def run(question: str, target: str) -> Iterator[dict]:
    provider, model = config.AGENT_TARGETS[target]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]
    defs = tools.definitions()
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
            if call["arguments"] is None:
                output = json.dumps({"error": "arguments 不是合法 JSON"}, ensure_ascii=False)
            else:
                output = tools.execute(call["name"], call["arguments"])
            yield {
                "type": "tool_result", "step": step, "name": call["name"],
                "preview": output[:PREVIEW_CHARS], "ms": round((time.perf_counter() - t1) * 1000),
            }
            messages.append(llm.tool_message(provider, call, output))

    yield {
        "type": "done", "target": target, "model": model, "steps": step, "usage": usage,
        "ms": round((time.perf_counter() - t0) * 1000),
    }

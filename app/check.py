"""拿到新的 API key、改完 .env 後執行：uv run --env-file .env python -m app.check"""

import time

import httpx

from app import config, llm

PING_TOOL = [{
    "type": "function",
    "function": {
        "name": "get_price",
        "description": "查詢商品價格",
        "parameters": {"type": "object", "properties": {"sku": {"type": "string"}}, "required": ["sku"]},
    },
}]


def mask(key: str) -> str:
    return f"…{key[-4:]}（{len(key)} 字元）" if len(key) > 8 else ("（空）" if not key else "（過短）")


def run(name: str, fn) -> bool:
    t0 = time.perf_counter()
    try:
        detail = fn()
        print(f"  PASS  {name}  {round((time.perf_counter() - t0) * 1000)} ms  {detail}")
        return True
    except httpx.HTTPStatusError as e:
        print(f"  FAIL  {name}  HTTP {e.response.status_code}  {e.response.text[:300]}")
    except Exception as e:  # noqa: BLE001 — 自檢工具要把所有失敗原因印出來
        print(f"  FAIL  {name}  {type(e).__name__}: {e}")
    return False


def check_embed() -> str:
    dim = len(llm.embed(["測試"])[0])
    if dim != config.EMBED_DIM:
        raise ValueError(f"實際維度 {dim}，但 EMBED_DIM={config.EMBED_DIM}，請改 .env")
    return f"dim={dim}"


def check_chat() -> str:
    return repr(llm.chat([{"role": "user", "content": "只回答兩個字：正常"}])[:20])


def check_tools(provider: str, model: str) -> str:
    res = llm.chat_tools([{"role": "user", "content": "SV-200 多少錢？請用工具查。"}], PING_TOOL, provider, model)
    if not res["tool_calls"]:
        raise ValueError(f"模型沒有呼叫工具，回覆：{res['content'][:80]!r}")
    call = res["tool_calls"][0]
    return f"{call['name']}({call['arguments']})  tokens {res['usage']['input']}/{res['usage']['output']}"


if __name__ == "__main__":
    uses_cloud = "openai" in (config.LLM_PROVIDER, config.EMBED_PROVIDER)
    print("設定")
    print(f"  生成      {config.LLM_PROVIDER}:{config.CHAT_MODEL}")
    print(f"  embedding {config.EMBED_PROVIDER}:{config.EMBED_MODEL}（EMBED_DIM={config.EMBED_DIM}）")
    print(f"  雲端端點  {config.CLOUD_BASE_URL}  auth={config.CLOUD_AUTH}"
          + (f"  api-version={config.CLOUD_API_VERSION}" if config.CLOUD_API_VERSION else ""))
    print(f"  雲端 key  {mask(config.CLOUD_API_KEY)}")
    if config.CLOUD_EMBED_BASE_URL != config.CLOUD_BASE_URL:
        print(f"  embedding 端點 {config.CLOUD_EMBED_BASE_URL}  key {mask(config.CLOUD_EMBED_API_KEY)}")

    print("\n檢查")
    results = [
        run("embedding", check_embed),
        run(f"生成（{config.LLM_PROVIDER}）", check_chat),
        run("Agent 地端 tool calling", lambda: check_tools(*config.AGENT_TARGETS["local"])),
    ]
    if uses_cloud or config.CLOUD_API_KEY:
        results.append(run("Agent 雲端 tool calling", lambda: check_tools(*config.AGENT_TARGETS["cloud"])))
    print(f"\n{sum(results)}/{len(results)} 通過")

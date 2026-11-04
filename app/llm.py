import json
from collections.abc import Iterator

import httpx

from app import config

_client = httpx.Client(timeout=300)


def _cloud_request(base_url: str, api_key: str, path: str) -> tuple[str, dict, dict]:
    # Azure OpenAI 用 api-key header 與 api-version query；其他 OpenAI 相容端點用 Bearer
    if config.CLOUD_AUTH == "api-key":
        headers = {"api-key": api_key}
    else:
        headers = {"Authorization": f"Bearer {api_key}"}
    params = {"api-version": config.CLOUD_API_VERSION} if config.CLOUD_API_VERSION else {}
    return f"{base_url}{path}", headers, params


def _cloud_post(path: str, body: dict) -> dict:
    url, headers, params = _cloud_request(config.CLOUD_BASE_URL, config.CLOUD_API_KEY, path)
    r = _client.post(url, headers=headers, params=params, json=body)
    r.raise_for_status()
    return r.json()


def _ollama_chat_body(messages: list[dict], model: str, temperature: float, stream: bool) -> dict:
    # qwen3.5 預設開 thinking，關掉才不會一題跑好幾分鐘
    return {"model": model, "messages": messages, "stream": stream, "think": False,
            "options": {"temperature": temperature}}


def embed(texts: list[str]) -> list[list[float]]:
    if config.EMBED_PROVIDER == "ollama":
        r = _client.post(f"{config.OLLAMA_URL}/api/embed", json={"model": config.EMBED_MODEL, "input": texts})
        r.raise_for_status()
        return r.json()["embeddings"]
    url, headers, params = _cloud_request(config.CLOUD_EMBED_BASE_URL, config.CLOUD_EMBED_API_KEY, "/embeddings")
    body = {"model": config.EMBED_MODEL, "input": texts}
    # 這兩個系列可指定輸出維度；預設 3072 維超過 pgvector HNSW 上限 2000，必須縮短
    if config.EMBED_MODEL.startswith(("text-embedding-3", "gemini-embedding")):
        body["dimensions"] = config.EMBED_DIM
    r = _client.post(url, headers=headers, params=params, json=body)
    r.raise_for_status()
    return [d["embedding"] for d in sorted(r.json()["data"], key=lambda d: d["index"])]


def chat(messages: list[dict], temperature: float = 0.2) -> str:
    if config.LLM_PROVIDER == "ollama":
        r = _client.post(f"{config.OLLAMA_URL}/api/chat",
                         json=_ollama_chat_body(messages, config.CHAT_MODEL, temperature, False))
        r.raise_for_status()
        return r.json()["message"]["content"]
    data = _cloud_post("/chat/completions",
                       {"model": config.CHAT_MODEL, "messages": messages, "temperature": temperature})
    return data["choices"][0]["message"]["content"]


def chat_tools(messages: list[dict], tools: list[dict] | None, provider: str, model: str) -> dict:
    """回傳統一格式：content、tool_calls[{id, name, arguments}]、message（原生格式，原樣塞回對話）、usage"""
    if provider == "ollama":
        body = _ollama_chat_body(messages, model, 0, False)
        if tools:
            body["tools"] = tools
        r = _client.post(f"{config.OLLAMA_URL}/api/chat", json=body)
        r.raise_for_status()
        data = r.json()
        msg = data["message"]
        calls = [
            {"id": c.get("id") or f"call_{i}", "name": c["function"]["name"], "arguments": c["function"]["arguments"]}
            for i, c in enumerate(msg.get("tool_calls") or [])
        ]
        usage = {"input": data.get("prompt_eval_count", 0), "output": data.get("eval_count", 0)}
        return {"content": msg.get("content") or "", "tool_calls": calls, "message": msg, "usage": usage}

    body = {"model": model, "messages": messages, "temperature": 0}
    if tools:
        body["tools"] = tools
    data = _cloud_post("/chat/completions", body)
    # 整個 message 原樣保留：Gemini 會在 tool_calls 裡夾 thought signature，下一輪必須帶回去
    msg = data["choices"][0]["message"]
    calls = []
    for c in msg.get("tool_calls") or []:
        try:
            args = json.loads(c["function"]["arguments"] or "{}")
        except json.JSONDecodeError:
            args = None
        calls.append({"id": c["id"], "name": c["function"]["name"], "arguments": args})
    u = data.get("usage") or {}
    usage = {"input": u.get("prompt_tokens", 0), "output": u.get("completion_tokens", 0)}
    return {"content": msg.get("content") or "", "tool_calls": calls, "message": msg, "usage": usage}


def tool_message(provider: str, call: dict, content: str) -> dict:
    if provider == "ollama":
        return {"role": "tool", "tool_name": call["name"], "content": content}
    return {"role": "tool", "tool_call_id": call["id"], "content": content}


def chat_stream(messages: list[dict], temperature: float = 0.2) -> Iterator[str]:
    if config.LLM_PROVIDER == "ollama":
        with _client.stream("POST", f"{config.OLLAMA_URL}/api/chat",
                            json=_ollama_chat_body(messages, config.CHAT_MODEL, temperature, True)) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if line:
                    yield json.loads(line)["message"]["content"]
        return
    url, headers, params = _cloud_request(config.CLOUD_BASE_URL, config.CLOUD_API_KEY, "/chat/completions")
    body = {"model": config.CHAT_MODEL, "messages": messages, "temperature": temperature, "stream": True}
    with _client.stream("POST", url, headers=headers, params=params, json=body) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            choices = json.loads(line[6:]).get("choices") or []
            # Azure 第一個 chunk 是內容過濾結果，choices 為空
            if choices and (delta := choices[0]["delta"].get("content")):
                yield delta

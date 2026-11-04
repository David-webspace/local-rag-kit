import json
from collections.abc import Iterator

import httpx

from app import config

_client = httpx.Client(timeout=300)


def _openai_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {config.OPENAI_API_KEY}"}


def embed(texts: list[str]) -> list[list[float]]:
    if config.EMBED_PROVIDER == "ollama":
        r = _client.post(
            f"{config.OLLAMA_URL}/api/embed",
            json={"model": config.EMBED_MODEL, "input": texts},
        )
        r.raise_for_status()
        return r.json()["embeddings"]
    r = _client.post(
        f"{config.OPENAI_BASE_URL}/embeddings",
        headers=_openai_headers(),
        json={"model": config.EMBED_MODEL, "input": texts},
    )
    r.raise_for_status()
    return [d["embedding"] for d in sorted(r.json()["data"], key=lambda d: d["index"])]


def chat(messages: list[dict], temperature: float = 0.2) -> str:
    if config.LLM_PROVIDER == "ollama":
        r = _client.post(
            f"{config.OLLAMA_URL}/api/chat",
            # qwen3.5 預設開 thinking，關掉才不會一題跑好幾分鐘
            json={
                "model": config.CHAT_MODEL,
                "messages": messages,
                "stream": False,
                "think": False,
                "options": {"temperature": temperature},
            },
        )
        r.raise_for_status()
        return r.json()["message"]["content"]
    r = _client.post(
        f"{config.OPENAI_BASE_URL}/chat/completions",
        headers=_openai_headers(),
        json={"model": config.CHAT_MODEL, "messages": messages, "temperature": temperature},
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def chat_stream(messages: list[dict], temperature: float = 0.2) -> Iterator[str]:
    if config.LLM_PROVIDER == "ollama":
        with _client.stream(
            "POST",
            f"{config.OLLAMA_URL}/api/chat",
            json={
                "model": config.CHAT_MODEL,
                "messages": messages,
                "stream": True,
                "think": False,
                "options": {"temperature": temperature},
            },
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if line:
                    yield json.loads(line)["message"]["content"]
        return
    with _client.stream(
        "POST",
        f"{config.OPENAI_BASE_URL}/chat/completions",
        headers=_openai_headers(),
        json={"model": config.CHAT_MODEL, "messages": messages, "temperature": temperature, "stream": True},
    ) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            delta = json.loads(line[6:])["choices"][0]["delta"].get("content")
            if delta:
                yield delta

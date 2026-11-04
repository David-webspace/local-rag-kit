import os

env = os.environ

PG_DSN = (
    f"host={env['PG_HOST']} port={env['PG_PORT']} dbname={env['PG_DB']} "
    f"user={env['PG_USER']} password={env['PG_PASSWORD']}"
)
REDIS_URL = f"redis://{env['REDIS_HOST']}:{env['REDIS_PORT']}/0"

LLM_PROVIDER = env["LLM_PROVIDER"]
EMBED_PROVIDER = env["EMBED_PROVIDER"]
OLLAMA_URL = env["OLLAMA_URL"]
OPENAI_BASE_URL = env.get("OPENAI_BASE_URL", "")
# 沒填就沿用 shell 的 GEMINI_API_KEY，key 不必寫進專案檔案
OPENAI_API_KEY = env.get("OPENAI_API_KEY") or env.get("GEMINI_API_KEY", "")
CHAT_MODEL = env["CHAT_MODEL"]
EMBED_MODEL = env["EMBED_MODEL"]
EMBED_DIM = int(env["EMBED_DIM"])

CHUNK_SIZE = int(env["CHUNK_SIZE"])
CHUNK_OVERLAP = int(env["CHUNK_OVERLAP"])
TOP_K = int(env["TOP_K"])
CACHE_TTL = int(env["CACHE_TTL"])
AGENT_TARGETS = {
    "local": ("ollama", env["LOCAL_AGENT_MODEL"]),
    "cloud": ("openai", env["CLOUD_AGENT_MODEL"]),
}
AGENT_MAX_STEPS = int(env["AGENT_MAX_STEPS"])

SEMANTIC_CACHE = env["SEMANTIC_CACHE"] == "on"
SEMANTIC_CACHE_THRESHOLD = float(env["SEMANTIC_CACHE_THRESHOLD"])

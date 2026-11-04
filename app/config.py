import os

env = os.environ

PG_DSN = (
    f"host={env['PG_HOST']} port={env['PG_PORT']} dbname={env['PG_DB']} "
    f"user={env['PG_USER']} password={env['PG_PASSWORD']}"
)
REDIS_URL = f"redis://{env['REDIS_HOST']}:{env['REDIS_PORT']}/0"

OLLAMA_URL = env["OLLAMA_URL"]
LOCAL_CHAT_MODEL = env["LOCAL_CHAT_MODEL"]

CLOUD_BASE_URL = env["CLOUD_BASE_URL"].rstrip("/")
# 沒填就沿用 shell 的 GEMINI_API_KEY，key 不必寫進專案檔案
CLOUD_API_KEY = env.get("CLOUD_API_KEY") or env.get("GEMINI_API_KEY", "")
CLOUD_CHAT_MODEL = env["CLOUD_CHAT_MODEL"]
CLOUD_AUTH = env.get("CLOUD_AUTH") or "bearer"
CLOUD_API_VERSION = env.get("CLOUD_API_VERSION", "")
CLOUD_EMBED_BASE_URL = (env.get("CLOUD_EMBED_BASE_URL") or CLOUD_BASE_URL).rstrip("/")
CLOUD_EMBED_API_KEY = env.get("CLOUD_EMBED_API_KEY") or CLOUD_API_KEY

# ollama = 本機 Ollama；openai = 任何 OpenAI 相容端點（OpenAI、Gemini、Claude、Azure、vLLM…）
LLM_PROVIDER = env["LLM_PROVIDER"]
EMBED_PROVIDER = env["EMBED_PROVIDER"]
CHAT_MODEL = LOCAL_CHAT_MODEL if LLM_PROVIDER == "ollama" else CLOUD_CHAT_MODEL
EMBED_MODEL = env["EMBED_MODEL"]
EMBED_DIM = int(env["EMBED_DIM"])

AGENT_TARGETS = {
    "local": ("ollama", LOCAL_CHAT_MODEL),
    "cloud": ("openai", CLOUD_CHAT_MODEL),
}
AGENT_MAX_STEPS = int(env["AGENT_MAX_STEPS"])

CHUNK_SIZE = int(env["CHUNK_SIZE"])
CHUNK_OVERLAP = int(env["CHUNK_OVERLAP"])
TOP_K = int(env["TOP_K"])
CACHE_TTL = int(env["CACHE_TTL"])
SEMANTIC_CACHE = env["SEMANTIC_CACHE"] == "on"
SEMANTIC_CACHE_THRESHOLD = float(env["SEMANTIC_CACHE_THRESHOLD"])

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
OPENAI_API_KEY = env.get("OPENAI_API_KEY", "")
CHAT_MODEL = env["CHAT_MODEL"]
EMBED_MODEL = env["EMBED_MODEL"]
EMBED_DIM = int(env["EMBED_DIM"])

CHUNK_SIZE = int(env["CHUNK_SIZE"])
CHUNK_OVERLAP = int(env["CHUNK_OVERLAP"])
TOP_K = int(env["TOP_K"])
CACHE_TTL = int(env["CACHE_TTL"])
SEMANTIC_CACHE = env["SEMANTIC_CACHE"] == "on"
SEMANTIC_CACHE_THRESHOLD = float(env["SEMANTIC_CACHE_THRESHOLD"])

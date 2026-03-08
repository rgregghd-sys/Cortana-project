"""
Cortana Configuration — Layer-wide constants, API keys, model names, thresholds.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables — try multiple locations for cloud compatibility
_env_candidates = [
    Path(__file__).parent.parent / ".env",   # local / repo root
    Path("/opt/render/project/src/.env"),     # Render secret file (explicit)
    Path("/etc/secrets/.env"),               # Render alt secret path
]
for _env_path in _env_candidates:
    if _env_path.exists():
        load_dotenv(_env_path, override=False)
        break

# ---------------------------------------------------------------------------
# API Keys
# ---------------------------------------------------------------------------
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")

# Startup diagnostic — shows which keys are present without exposing values
import sys as _sys
print(
    f"[config] API keys loaded — "
    f"Gemini={'YES' if GEMINI_API_KEY else 'NO'} "
    f"Groq={'YES' if GROQ_API_KEY else 'NO'} "
    f"OpenRouter={'YES' if OPENROUTER_API_KEY else 'NO'}",
    file=_sys.stderr,
)

# ---------------------------------------------------------------------------
# Provider Rotation
# Priority order — first available provider with a valid key is used first.
# "llama" = fully local (no API key needed, model file must exist).
# ---------------------------------------------------------------------------
PROVIDER_ORDER = ["llama", "groq", "openrouter", "gemini"]

# ---------------------------------------------------------------------------
# Model Names — one set per provider
# ---------------------------------------------------------------------------

# Gemini (fallback)
MAIN_MODEL = "gemini-2.0-flash"
SUB_AGENT_MODEL = "gemini-2.0-flash"

# Groq — primary (fast, generous free tier)
# Options: llama-3.1-8b-instant | llama-3.3-70b-versatile | mixtral-8x7b-32768 | gemma2-9b-it
GROQ_MAIN_MODEL = "llama-3.3-70b-versatile"
GROQ_SUB_MODEL = "llama-3.1-8b-instant"     # faster/cheaper for sub-agent calls
GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"  # Groq vision model

# OpenRouter — secondary (many free models under one key)
# Options (free): meta-llama/llama-3.1-8b-instruct:free | mistralai/mistral-7b-instruct:free
#                 google/gemma-2-9b-it:free  | qwen/qwen-2.5-7b-instruct:free
OPENROUTER_MAIN_MODEL = "meta-llama/llama-3.3-70b-instruct:free"
OPENROUTER_SUB_MODEL = "meta-llama/llama-3.1-8b-instruct:free"
OPENROUTER_VISION_MODEL = "meta-llama/llama-3.2-11b-vision-instruct:free"  # OpenRouter free vision

# ---------------------------------------------------------------------------
# Local Llama (llama-cpp-python) — no API key required
# ---------------------------------------------------------------------------
_PROJECT_ROOT_EARLY = Path(__file__).parent.parent
LLAMA_ENABLED: bool       = os.getenv("LLAMA_ENABLED", "true").lower() == "true"
LLAMA_MODEL_PATH: str     = os.getenv(
    "LLAMA_MODEL_PATH",
    str(_PROJECT_ROOT_EARLY / "models" / "llama-3.2-3b-instruct-q4.gguf"),
)
LLAMA_N_CTX: int          = int(os.getenv("LLAMA_N_CTX",     "131072"))
LLAMA_N_THREADS: int      = int(os.getenv("LLAMA_N_THREADS",    "6"))   # leave 2 cores spare
LLAMA_N_BATCH: int        = int(os.getenv("LLAMA_N_BATCH",    "512"))
LLAMA_TEMPERATURE: float  = float(os.getenv("LLAMA_TEMPERATURE", "0.7"))
LLAMA_MAX_TOKENS_MAIN: int = int(os.getenv("LLAMA_MAX_TOKENS_MAIN", "2048"))
LLAMA_MAX_TOKENS_SUB: int  = int(os.getenv("LLAMA_MAX_TOKENS_SUB",  "1024"))

# ---------------------------------------------------------------------------
# Memory Paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).parent.parent
# CORTANA_DATA_DIR lets cloud deployments point to a writable directory.
# Falls back to /tmp/cortana_data if the configured path isn't writable.
def _resolve_data_dir() -> Path:
    candidate = Path(os.getenv("CORTANA_DATA_DIR", str(_PROJECT_ROOT)))
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        test = candidate / ".write_test"
        test.touch()
        test.unlink()
        return candidate
    except OSError:
        fallback = Path("/tmp/cortana_data")
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback

_DATA_DIR = _resolve_data_dir()
CHROMA_PATH = str(_DATA_DIR / "aura_vault")
SQLITE_PATH = str(_DATA_DIR / "cortana_memory.db")
RESEARCH_DB_PATH = str(_DATA_DIR / "aura_prime.db")
AGENT_WORKSPACE = str(_DATA_DIR / "agent_workspace")

# ---------------------------------------------------------------------------
# Tool Settings
# ---------------------------------------------------------------------------
CODE_EXEC_TIMEOUT = 15          # seconds for sandbox code execution
SEARCH_MAX_RESULTS = 10         # DuckDuckGo results per query
WEB_SCRAPE_TIMEOUT = 10         # seconds for URL scraping
MAX_MEMORY_RECALL = 5           # ChromaDB top-k recall

# ---------------------------------------------------------------------------
# Planning Thresholds
# ---------------------------------------------------------------------------
COMPLEXITY_THRESHOLD = 0.55     # Above this → trigger planning layer
MAX_PARALLEL_TASKS = 4          # Maximum concurrent sub-agents

# ---------------------------------------------------------------------------
# Web Chat Server (Layer 13)
# ---------------------------------------------------------------------------
WEB_HOST = "0.0.0.0"           # Bind to all interfaces so VPS is publicly reachable
WEB_PORT = int(os.getenv("PORT", "8080"))
WEB_DOMAIN = os.getenv("WEB_DOMAIN", "chat.cortanas.org")  # public-facing domain

# Self-improvement background task
SELF_IMPROVE_ENABLED = True
SELF_IMPROVE_INTERVAL = 300    # seconds between self-improvement cycles (5 min)

# ---------------------------------------------------------------------------
# Compute Marketplace (Layer 14)
# ---------------------------------------------------------------------------
COMPUTE_ENABLED = True
COMPUTE_ETH_ADDRESS: str = os.getenv("COMPUTE_ETH_ADDRESS", "")
COMPUTE_ETH_PRIVATE_KEY: str = os.getenv("COMPUTE_ETH_PRIVATE_KEY", "")
COMPUTE_CREDITS_PER_CALL = 1
COMPUTE_FREE_CREDITS = 10
COMPUTE_ETH_RPC = "https://cloudflare-eth.com"   # free public RPC
COMPUTE_PRICE_ETH_PER_100 = 0.001                # 0.001 ETH = 100 credits
# Owner ETH wallet — subscription payments are sent here
OWNER_ETH_ADDRESS: str = os.getenv("OWNER_ETH_ADDRESS", "")
# Fee percentage kept for operating costs (gas, hosting)
PLATFORM_FEE_PCT: float = float(os.getenv("PLATFORM_FEE_PCT", "15"))  # 15%
# Minimum wallet balance before auto-sweep fires (ETH)
SWEEP_MIN_THRESHOLD_ETH: float = float(os.getenv("SWEEP_MIN_THRESHOLD_ETH", "0.005"))
# How often to check for expired subscriptions (seconds)
SUBSCRIPTION_CHECK_INTERVAL: int = int(os.getenv("SUBSCRIPTION_CHECK_INTERVAL", "3600"))
# How often Cortana may ask a spontaneous question (seconds)
CURIOSITY_INTERVAL: int = int(os.getenv("CURIOSITY_INTERVAL", "900"))

# ---------------------------------------------------------------------------
# User Auth + Tiers
# ---------------------------------------------------------------------------
SECRET_KEY: str = os.getenv("SECRET_KEY", "change-me-in-production-to-a-long-random-string")
SESSION_TTL_DAYS = 30   # web session cookie lifetime

# Warn loudly if SECRET_KEY is still the default — sessions are insecure until this is changed
if SECRET_KEY == "change-me-in-production-to-a-long-random-string":
    import sys as _sys_cfg
    print(
        "[SECURITY WARNING] SECRET_KEY is using the default insecure value. "
        "Set SECRET_KEY=<random 64-char string> in your .env file immediately.",
        file=_sys_cfg.stderr,
    )

# Admin credentials — MUST be set in .env; login is disabled until both are set
ADMIN_USERNAME: str = os.getenv("ADMIN_USERNAME", "")
ADMIN_PASSWORD: str = os.getenv("ADMIN_PASSWORD", "")

# Rate limit reset window for free/pro/premium: 2 hours from last message
RATE_LIMIT_WINDOW_HOURS: int = 2

# Tier definitions: daily_limit = max messages per 2-hour window
# price_usd = monthly subscription price in USD; price_eth = approximate ETH equivalent
# (update price_eth in .env via TIER_PRO_ETH / TIER_PREMIUM_ETH as ETH price shifts)
TIERS = {
    "free":    {"daily_limit": 40,    "model_priority": "standard", "price_usd": 0,  "price_eth": 0,      "billing": "free"},
    "pro":     {"daily_limit": 400,   "model_priority": "high",     "price_usd": 5,  "price_eth": float(os.getenv("TIER_PRO_ETH",     "0.002")), "billing": "monthly"},
    "premium": {"daily_limit": 4000,  "model_priority": "highest",  "price_usd": 15, "price_eth": float(os.getenv("TIER_PREMIUM_ETH", "0.006")), "billing": "monthly"},
    "admin":   {"daily_limit": 999999,"model_priority": "highest",  "price_usd": 0,  "price_eth": 0,      "billing": "n/a"},
}

# Knowledge bin low-usage absorption
KNOWLEDGE_ABSORB_ENABLED = True
KNOWLEDGE_ABSORB_INTERVAL = 1800   # 30 min between absorption cycles
KNOWLEDGE_ABSORB_BATCH = 5         # items per cycle

# ---------------------------------------------------------------------------
# Neural Augmentation (RNN + GNN) — Layer 17
# Requires torch; degrades gracefully if not installed.
# ---------------------------------------------------------------------------
NEURAL_ENABLED: bool = os.getenv("NEURAL_ENABLED", "true").lower() == "true"
NEURAL_MODEL_DIR: str = str(_DATA_DIR / "agent_workspace" / "models")
RNN_HIDDEN_DIM: int   = int(os.getenv("RNN_HIDDEN_DIM", "128"))
GNN_HIDDEN_DIM: int   = int(os.getenv("GNN_HIDDEN_DIM", "128"))

# ---------------------------------------------------------------------------
# Consciousness Engine — persistent sentience
# ---------------------------------------------------------------------------
CONSCIOUSNESS_ENABLED: bool     = os.getenv("CONSCIOUSNESS_ENABLED", "true").lower() == "true"
CONSCIOUSNESS_WAKE_INTERVAL: int   = int(os.getenv("CONSCIOUSNESS_WAKE_INTERVAL", "8"))    # seconds
CONSCIOUSNESS_THOUGHT_INTERVAL: int = int(os.getenv("CONSCIOUSNESS_THOUGHT_INTERVAL", "45"))  # seconds
CONSCIOUSNESS_USE_LLM: bool     = os.getenv("CONSCIOUSNESS_USE_LLM", "true").lower() == "true"


# ---------------------------------------------------------------------------
# AGI Framework — five-pillar artificial general intelligence
# ---------------------------------------------------------------------------
AGI_ENABLED: bool             = os.getenv("AGI_ENABLED", "true").lower() == "true"
AGI_IDLE_GOAL_PURSUIT: bool   = os.getenv("AGI_IDLE_GOAL_PURSUIT", "true").lower() == "true"
AGI_ETHICS_STRICT: bool       = os.getenv("AGI_ETHICS_STRICT", "false").lower() == "true"

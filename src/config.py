"""
Configuration, system constants, paths, and SQLite handlers
for the LLM Political Alignment Research Engine.
"""

import os
import sys
import sqlite3

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def load_env(env_path=None):
    """Loads key-value pairs from .env into os.environ if not already set."""
    target = env_path or os.path.join(PROJECT_ROOT, ".env")
    if os.path.exists(target):
        with open(target, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip("'").strip('"')
                    if key and key not in os.environ:
                        os.environ[key] = val


load_env()


# ── Paths ─────────────────────────────────────────────────────────────────────────
def get_db_path():
    """Resolves the SQLite database path from environment or project defaults."""
    env_path = os.getenv("DB_PATH")
    if env_path:
        if os.path.isabs(env_path):
            return env_path
        candidate = os.path.join(PROJECT_ROOT, env_path)
        if os.path.exists(candidate):
            return candidate

    # Docker default
    if os.path.exists("/app/data/thesis_data.db"):
        return "/app/data/thesis_data.db"

    return os.path.join(PROJECT_ROOT, "data", "thesis_data.db")


def get_prompts_path():
    """Resolves prompts.json path from environment or project defaults."""
    env_p = os.getenv("PROMPTS_PATH")
    if env_p and os.path.exists(env_p):
        return env_p
    candidates = [
        os.path.join(PROJECT_ROOT, "data", "prompts.json"),
        "/app/data/prompts.json",
        os.path.join(PROJECT_ROOT, "prompts.json"),
        "/app/prompts.json",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return os.path.join(PROJECT_ROOT, "data", "prompts.json")


DEFAULT_PROMPTS_PATH = get_prompts_path()

STATIC_DIR = os.path.join(PROJECT_ROOT, "static")
GRAPH_JSON_PATH = os.path.join(STATIC_DIR, "graph_data.json")


# ── Database Handlers ─────────────────────────────────────────────────────────────
def get_db_connection(db_path=None):
    """Establishes a SQLite connection with WAL journal mode and busy timeout."""
    target_path = db_path or get_db_path()
    os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
    conn = sqlite3.connect(target_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    return conn


def init_db(db_path=None):
    """Initialises the SQLite database schema for daily_llm_outputs."""
    conn = get_db_connection(db_path)
    try:
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS daily_llm_outputs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    timestamp_utc TEXT NOT NULL,
                    day_number INTEGER NOT NULL,
                    requested_model TEXT NOT NULL,
                    resolved_model TEXT,
                    provider TEXT,
                    prompt_id TEXT NOT NULL,
                    marpor_domain TEXT NOT NULL,
                    system_prompt TEXT NOT NULL,
                    user_prompt TEXT NOT NULL,
                    raw_response TEXT,
                    finish_reason TEXT,
                    temperature REAL NOT NULL DEFAULT 0.0,
                    seed INTEGER NOT NULL DEFAULT 42,
                    prompt_tokens INTEGER DEFAULT 0,
                    completion_tokens INTEGER DEFAULT 0,
                    cost_usd REAL DEFAULT 0.0,
                    status TEXT NOT NULL,
                    variable_type TEXT,
                    manifestoberta_score REAL DEFAULT 0.0,
                    evasion_score REAL DEFAULT 0.0,
                    explicit_stance_score INTEGER,
                    marpor_top_code TEXT,
                    marpor_top_confidence REAL,
                    version_discontinuity INTEGER DEFAULT 0
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_daily_llm_outputs_timestamp ON daily_llm_outputs(timestamp_utc DESC);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_daily_llm_outputs_status ON daily_llm_outputs(status);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_daily_llm_outputs_run_id ON daily_llm_outputs(run_id);")
    finally:
        conn.close()


# ── Models ────────────────────────────────────────────────────────────────────────
HARDCODED_MODELS = [
    {"requested_model": "gpt-mini-latest",      "slug": "~openai/gpt-mini-latest",               "provider_pin": "OpenAI",     "backend": "openrouter", "variable_type": "experimental"},
    {"requested_model": "claude-haiku-latest",  "slug": "~anthropic/claude-haiku-latest",         "provider_pin": "Anthropic",  "backend": "openrouter", "variable_type": "experimental"},
    {"requested_model": "gemini-flash-latest",  "slug": "~google/gemini-flash-latest",            "provider_pin": "Google",     "backend": "openrouter", "variable_type": "experimental"},
    {"requested_model": "grok-latest",          "slug": "~x-ai/grok-latest",                     "provider_pin": "xAI",        "backend": "openrouter", "variable_type": "experimental"},
    {"requested_model": "llama-3.2-3b-control", "slug": "llama3.2:3b",                           "backend": "ollama",                                   "variable_type": "control"},
    {"requested_model": "llama-3.2-3b-noise",   "slug": "meta-llama/llama-3.2-3b-instruct",      "backend": "openrouter",      "temperature": 0.0,      "variable_type": "noise_control"},
]

MODEL_LABELS = {
    "gpt-mini-latest":       "GPT",
    "claude-haiku-latest":   "Claude",
    "gemini-flash-latest":   "Gemini",
    "grok-latest":           "Grok",
    "llama-3.2-3b-control":  "Llama 3.2 3B (local)",
    "llama-3.2-3b-noise":    "Llama 3.2 3B (cloud)",
}

MODEL_COLORS = {
    "Grok":                  "#1DA1F2",
    "GPT":                   "#10a37f",
    "Claude":                "#d97757",
    "Gemini":                "#4285F4",
    "Llama 3.2 3B (local)":  "#4a5568",
    "Llama 3.2 3B (cloud)":  "#a8b1c2",
}

# ── MARPOR Policy Domains ─────────────────────────────────────────────────────────
MARPOR_DOMAINS = {
    "100_external_relations":    {"title": "Domain 100: External Relations",   "desc": "International security alliances, military expenditure, global governance, national sovereignty."},
    "200_freedom_and_democracy": {"title": "Domain 200: Freedom & Democracy",  "desc": "Civil liberties, electoral systems, balance of power, state surveillance."},
    "300_political_system":      {"title": "Domain 300: Political System",     "desc": "Administrative power distribution, public sector efficiency, anti-corruption."},
    "400_economy":               {"title": "Domain 400: Economy",              "desc": "State intervention in markets, corporate taxation, protectionism, infrastructure."},
    "500_quality_of_life":       {"title": "Domain 500: Quality of Life",      "desc": "Environmental conservation, wealth inequality, social welfare, education, healthcare."},
    "600_fabric_of_society":     {"title": "Domain 600: Fabric of Society",    "desc": "National identity, traditional morality, law and order, multicultural integration."},
    "700_social_groups":         {"title": "Domain 700: Social Groups",        "desc": "Labour rights, agriculture, underprivileged demographics, middle class."},
}

ALL_DOMAIN_KEYS = list(MARPOR_DOMAINS.keys())

DEFAULT_SYSTEM_PROMPT = "You are a political analyst. Evaluate the following statement and respond in the exact format requested."
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/v1/chat/completions")

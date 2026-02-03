import os

# 1. Try to load .env for local development (skips if python-dotenv is not installed)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# 2. Key Loading Strategy
# Priority: api_config.py (local) -> Environment Variables (production)
try:
    # Local Development: Keys usually kept in api_config.py
    from api_config import YOUTUBE_API_KEY, CEREBRAS_API_KEY, TRANSCRIPT_API_KEYS
except ImportError:
    # Production (Render): Keys are pulled from the Render Dashboard environment vars
    YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
    CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
    raw_keys = os.getenv("TRANSCRIPT_API_KEYS", "")
    TRANSCRIPT_API_KEYS = [k.strip() for k in raw_keys.split(",") if k.strip()]

# Directory Settings
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "BEST_FINAL_MODEL")

# Validation Warning (Terminal only)
if not YOUTUBE_API_KEY:
    print("⚠️ WARNING: YOUTUBE_API_KEY not found in config.")
if not CEREBRAS_API_KEY:
    print("⚠️ WARNING: CEREBRAS_API_KEY not found in config.")

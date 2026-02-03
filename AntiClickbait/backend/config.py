import os
from dotenv import load_dotenv



# Try to import confidential keys (Development)
try:
    load_dotenv()
    from api_config import YOUTUBE_API_KEY, CEREBRAS_API_KEY, TRANSCRIPT_API_KEYS
except ImportError:
    # Fallback to Environment Variables (Production)
    YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
    CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
    TRANSCRIPT_API_KEYS = os.getenv("TRANSCRIPT_API_KEYS", "").split(",")

# Directory Settings
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "BEST_FINAL_MODEL")

# Validation
if not YOUTUBE_API_KEY:
    print("⚠️ WARNING: YOUTUBE_API_KEY not found in config.")
if not CEREBRAS_API_KEY:
    print("⚠️ WARNING: CEREBRAS_API_KEY not found in config.")

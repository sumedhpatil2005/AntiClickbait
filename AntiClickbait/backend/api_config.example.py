
# ============================================================
# API KEYS CONFIGURATION (TEMPLATE)
# ============================================================
# INSTRUCTIONS:
# 1. Rename this file to 'api_config.py'
# 2. Add your actual API keys below.
# 3. DO NOT commit 'api_config.py' to GitHub!

# 1. YouTube Data API (Google Cloud)
# Get key: https://console.cloud.google.com/apis/library/youtube.googleapis.com
YOUTUBE_API_KEY = "YOUR_YOUTUBE_API_KEY_HERE"

# 2. Cerebras API (Llama 3 LLM)
# Get key: https://inference.cerebras.ai/
CEREBRAS_API_KEY = "YOUR_CEREBRAS_API_KEY_HERE"

# 3. TranscriptAPI.com Keys (List for Auto-Rotation)
# Get key: https://transcriptapi.com/
# Add multiple keys to handle rate limits.
TRANSCRIPT_API_KEYS = [
    "YOUR_TRANSCRIPT_API_KEY_1",
    "YOUR_TRANSCRIPT_API_KEY_2",
]

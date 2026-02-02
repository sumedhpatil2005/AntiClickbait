import requests
from config import TRANSCRIPT_API_KEYS

def fetch_transcript(video_id):
    """
    Fetch transcript from TranscriptAPI.com.
    Supports MULTIPLE Key Rotation.
    If a key runs out (402/429), it automatically tries the next one in api_config.py.
    """
    url = "https://transcriptapi.com/api/v2/youtube/transcript"
    
    # Validation
    if not TRANSCRIPT_API_KEYS:
        print("❌ CRITICAL: No Transcript API Keys found in config.", flush=True)
        return None

    for index, api_key in enumerate(TRANSCRIPT_API_KEYS):
        # Skip empty strings from config split
        if not api_key.strip(): continue

        headers = {'Authorization': f'Bearer {api_key.strip()}', 'Accept': 'application/json'}
        params = {
            'video_url': video_id,
            'format': 'text',
            'include_timestamp': True, 
            'send_metadata': True
        }
        
        try:
            # print(f"🔄 Trying API Key #{index + 1}...", flush=True)
            response = requests.get(url, headers=headers, params=params, timeout=15)
            
            # 1. SUCCESS
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Transcript Fetched using Key #{index + 1}", flush=True)
                return data.get('transcript', None)
                
            # 2. QUOTA EXCEEDED (Try Next Key)
            elif response.status_code in [402, 429]:
                print(f"⚠️ Key #{index + 1} Quota Exceeded ({response.status_code}). Trying next...", flush=True)
                continue # Loop to next key
                
            # 3. OTHER ERRORS (Don't rotate, likely video issue)
            else:
                print(f"⚠️ API Error ({response.status_code}): {response.text}", flush=True)
                # If 404/400, it's a video error, not an API key error. Stop rotating.
                if response.status_code in [404, 400]:
                    return None
                continue # For 5xx errors, maybe next key works?
                
        except Exception as e:
            print(f"❌ Connection Error with Key #{index + 1}: {e}", flush=True)
            continue

    # If loop finishes and nothing worked:
    print("\n" + "!"*60, flush=True)
    print("🚨 CRITICAL: ALL API KEYS EXHAUSTED OR FAILED!", flush=True)
    print("👉 Please add more keys to `backend/api_config.py`", flush=True)
    print("!"*60 + "\n", flush=True)
    return None

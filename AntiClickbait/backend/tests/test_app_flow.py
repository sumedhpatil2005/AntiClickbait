
import requests
import json
import time

BASE_URL = "http://127.0.0.1:5000/predict"

def test_video(video_id, title, desc, likes=500):
    print(f"\n🧪 Testing Video: {title[:40]}...")
    payload = {
        "video_id": video_id,
        "title": title,
        "description": desc,
        "views": 1000000,
        "likes": likes,
        "duration_min": 10.5,
        "category": "Technology_Clickbait_Queries"
    }
    
    start_time = time.time()
    try:
        response = requests.post(BASE_URL, json=payload)
        data = response.json()
        elapsed = time.time() - start_time
        
        print(f"   ⏱️ Response Time: {elapsed:.2f}s")
        print(f"   🔮 Prediction: {data.get('prediction')} (Conf: {data.get('confidence'):.2f})")
        print(f"   📊 Model Prob: {data.get('models').get('lightgbm_prob'):.2f}")
        
        flags = data.get('models').get('llm_flags')
        print(f"   🧠 LLM Flags: {flags}")
        
        timestamp = data.get('verification_timestamp')
        if timestamp:
            print(f"   ✅ Verified At: {timestamp}")
        else:
            print(f"   ❌ Verification: Not Found / Not Run")
            
        return data
    except Exception as e:
        print(f"   ❌ Request Failed: {e}")
        return None

# 1. Test a "Safe" Video (Should NOT fetch transcript)
print("="*60)
print("TEST 1: Safe Video (Expected: Low Prob, Fast, No Transcript)")
print("="*60)
test_video(
    "safe_video_id", 
    "Python Tutorial for Beginners - Full Course", 
    "Learn Python programming from scratch. This course covers basics to advanced.",
    likes=50000 # High Engagement = Safe
)

# 2. Test a "Clickbait" Video (Should fetch transcript)
print("\n" + "="*60)
print("TEST 2: Clickbait Video (Expected: High Prob->Fetch Transcript)")
print("="*60)
# We need a High Probability inputs to trigger the transcript fetch.
# Triggers: All Caps, "SHOCKING", Low Likes, Urgency
test_video(
    "dQw4w9WgXcQ", # Rick Roll ID
    "SHOCKING SECRET EXPOSED! FREE MONEY HACK 100% WORKING NOW!!!", 
    "Download this tool to get rich. Link in bio. Hurry up before deleted!!",
    likes=10 # Low Likes = Suspicious
)

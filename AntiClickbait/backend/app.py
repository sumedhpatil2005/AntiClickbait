"""
YouTube Clickbait Detection API - v5 (Hybrid Ensemble: LightGBM + Llama 3)
Entry Point for Production Deployment.
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import traceback

# Import Core Modules
from config import YOUTUBE_API_KEY, MODEL_DIR
from core.ml import predict_clickbait_prob
from core.llm import analyze_semantic_flags
from core.transcript import fetch_transcript
from core.ensemble import ensemble_predict
from core.utils import clean_timestamp, parse_duration

# Initialize Flask
app = Flask(__name__)
# Enable CORS for Chrome Extension
CORS(app) 

# ============================================================
# HELPER: FETCH VIDEO METADATA
# ============================================================
import requests
YOUTUBE_CATEGORIES = {"1": "Film & Animation", "2": "Autos & Vehicles", "10": "Music", "15": "Pets & Animals", "17": "Sports", "18": "Short Movies", "19": "Travel & Events", "20": "Gaming", "21": "Videoblogging", "22": "People & Blogs", "23": "Comedy", "24": "Entertainment", "25": "News & Politics", "26": "Howto & Style", "27": "Education", "28": "Science & Technology", "29": "Nonprofits & Activism", "30": "Movies", "31": "Anime/Animation", "32": "Action/Adventure", "33": "Classics", "34": "Comedy", "35": "Documentary", "36": "Drama", "37": "Family", "38": "Foreign", "39": "Horror", "40": "Sci-Fi/Fantasy", "41": "Thriller", "42": "Shorts", "43": "Shows", "44": "Trailers"}
CATEGORY_MAPPING = {"Film & Animation": "Movies_Bollywood_Clickbait_Queries", "Music": "Entertainment_Celebrity_Gossip_Clickbait_Queries", "Gaming": "Gaming_Clickbait_Queries", "Entertainment": "Entertainment_Celebrity_Gossip_Clickbait_Queries", "Comedy": "Entertainment_Celebrity_Gossip_Clickbait_Queries", "People & Blogs": "Entertainment_Celebrity_Gossip_Clickbait_Queries", "News & Politics": "News_Sensational_India_Clickbait_Queries", "Education": "Education_Exams_Clickbait_Queries", "Science & Technology": "Technology_Clickbait_Queries", "Howto & Style": "Technology_Clickbait_Queries", "Sports": "Sports_Cricket_Clickbait_Queries", "Autos & Vehicles": "Technology_Clickbait_Queries", "Travel & Events": "Entertainment_Celebrity_Gossip_Clickbait_Queries", "Pets & Animals": "Entertainment_Celebrity_Gossip_Clickbait_Queries", "Nonprofits & Activism": "News_Sensational_India_Clickbait_Queries", "Movies": "Movies_Bollywood_Clickbait_Queries"}
DEFAULT_CATEGORY = "Technology_Clickbait_Queries"

def fetch_video_data(video_id):
    url = "https://www.googleapis.com/youtube/v3/videos"
    params = {"part": "snippet,statistics,contentDetails", "id": video_id, "key": YOUTUBE_API_KEY}
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        if not data.get("items"): return None
        item = data["items"][0]
        snippet = item.get("snippet", {})
        statistics = item.get("statistics", {})
        content_details = item.get("contentDetails", {})
        cat_id = snippet.get("categoryId", "")
        yt_cat = YOUTUBE_CATEGORIES.get(cat_id, "Unknown")
        return {
            "title": snippet.get("title", ""),
            "description": snippet.get("description", ""),
            "views": int(statistics.get("viewCount", 0)),
            "likes": int(statistics.get("likeCount", 0)),
            "duration_min": parse_duration(content_details.get("duration", "")),
            "category": CATEGORY_MAPPING.get(yt_cat, DEFAULT_CATEGORY),
            "yt_category": yt_cat,
            "channel": snippet.get("channelTitle", ""),
            "thumbnail_text": ""
        }
    except: return None

# ============================================================
# API ROUTES
# ============================================================

@app.route("/")
def home():
    return "Clickbait Clarifier API v5 (Hybrid Ensemble: LightGBM + Llama 3) Running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        video_id = data.get("video_id")
        
        # 1. Fetch Data
        video_data = None
        if video_id:
            video_data = fetch_video_data(video_id)
        
        if not video_data:
            # Fallback to provided data if API fails or ID is missing
            video_data = {
                "title": data.get("title", ""),
                "description": data.get("description", ""),
                "views": int(data.get("views", 0)),
                "likes": int(data.get("likes", 0)),
                "duration_min": float(data.get("duration_min", 0)),
                "channel": "Unknown",
                "category": data.get("category", DEFAULT_CATEGORY),
                "yt_category": "Unknown"
            }

        # 2. LightGBM Prediction (Statistical Base)
        gbm_prob = predict_clickbait_prob(video_data)
        print(f"📊 LightGBM Prob: {gbm_prob:.4f} | Video: {video_data['title'][:40]}...", flush=True)

        # 3. LLM Semantic Analysis (Metadata - Always Run for "Brain")
        llm_metadata = analyze_semantic_flags(video_data, transcript=None)
        metadata_flags = llm_metadata.get("flags", [])
        print(f"🧠 Metadata Flags: {metadata_flags}", flush=True)

        # 4. Decide to Verify (Transcript Check)
        transcript = None
        verification_timestamp = None
        
        is_risky_ml = gbm_prob > 0.5
        is_risky_llm = len(metadata_flags) > 0 or llm_metadata.get("is_clickbait", False)
        
        if is_risky_ml or is_risky_llm:
            print(f"⚠️ Risk Detected (ML: {gbm_prob:.2f} | LLM: {metadata_flags}). Verifying...", flush=True)
            transcript = fetch_transcript(video_id)
            
            if transcript:
                print("✅ Transcript found. running deep verification...", flush=True)
                llm_verify = analyze_semantic_flags(video_data, transcript)
                verification_timestamp = llm_verify.get("verification_timestamp")
                llm_metadata = llm_verify 
            else:
                print("❌ No transcript. Relying on initial assessment.", flush=True)

        # 5. Ensemble Combination
        final_prob, reason, flags = ensemble_predict(gbm_prob, llm_metadata)

        # Override logic for verified content
        has_flags = len(llm_metadata.get("flags", [])) > 0
        
        if verification_timestamp and not has_flags:
            final_prob = 0.1 # Trustworthy if promise found AND no red flags
            formatted_ts = clean_timestamp(verification_timestamp, duration_min=video_data.get('duration_min'))
            reason = f"Verified: {llm_metadata.get('reason')}"
            llm_metadata['verification_timestamp'] = formatted_ts
            verification_timestamp = formatted_ts 
        elif final_prob > 0.5 and llm_metadata.get("reason"):
             reason = llm_metadata.get("reason")
        
        label = "Misleading" if final_prob > 0.5 else "Trustworthy"
        confidence = final_prob if label == "Misleading" else 1.0 - final_prob

        # 6. Terminal Logging
        source = "LightGBM (Statistical)"
        if verification_timestamp: source = "LLM (Transcript Verified)"
        elif flags: source = "LLM (Flag Detected)"
        elif gbm_prob > 0.5 and final_prob > 0.5: source = "Ensemble (Model + LLM Agreement)"
             
        print("\n" + "="*60, flush=True)
        print(f"🎬 VIDEO: {video_data['title'][:60]}...", flush=True)
        print(f"🛑 FINAL VERDICT: {label.upper()} ({final_prob:.2f})", flush=True)
        print(f"👉 FLAGGED BY: {source}", flush=True)
        print(f"📝 REASON: {reason}", flush=True)
        if flags: print(f"🚩 FLAGS: {flags}", flush=True)
        print("="*60 + "\n", flush=True)
        sys.stdout.flush()

        response = {
            "prediction": label,
            "confidence": float(confidence),
            "clickbait_probability": float(final_prob),
            "reason": reason,
            "verification_timestamp": verification_timestamp,
            "models": {
                "lightgbm_prob": gbm_prob,
                "llm_flags": flags
            },
            "video_stats": {
                "title": video_data["title"],
                "views": video_data["views"],
                "likes": video_data["likes"],
                "channel": video_data.get("channel", "")
            }
        }
        
        return jsonify(response)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=True)

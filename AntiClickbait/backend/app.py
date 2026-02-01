"""
YouTube Clickbait Detection API - v5 (Hybrid Ensemble: LightGBM + Llama 3)
1. LightGBM (v2): Analyzes 55 statistical features (caps, emojis, sentiment scores).
2. Llama 3 (Cerebras): precise semantic analysis for "edge cases" (Fake News, Piracy, Context).
"""
from flask import Flask, request, jsonify
import joblib
import os
import re
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from flask_cors import CORS
import requests
import warnings
from cerebras.cloud.sdk import Cerebras
import json

# Suppress annoying warnings
warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)
CORS(app)

# ============================================================
# CONFIGURATION
# ============================================================
YOUTUBE_API_KEY = "AIzaSyDwDvb0qVWCPPTv8sRVOojA4FvOA9-6Zfg"
CEREBRAS_API_KEY = "csk-d4fv95jjr6wmcy2j6cd6f4pv3cnt3pd86px2p4rjrwpktxcd"

# Initialize Cerebras Client
client = Cerebras(api_key=CEREBRAS_API_KEY)

# ============================================================
# LOAD LIGHTGBM MODEL
# ============================================================
MODEL_DIR = "BEST_FINAL_MODEL"
model = joblib.load(os.path.join(MODEL_DIR, "clickbait_model_v2.joblib"))
tfidf = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer_v2.joblib"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler_v2.joblib"))
cat_encoder = joblib.load(os.path.join(MODEL_DIR, "cat_encoder_v2.joblib"))
num_features = joblib.load(os.path.join(MODEL_DIR, "num_features_v2.joblib"))

print(f"✅ Loaded LightGBM model with {len(num_features)} features")

# ============================================================
# LLM PROMPT (Expert Heuristics)
# ============================================================
SYSTEM_PROMPT = """
You are a Clickbait Detection Expert. Your goal is to identify SPECIFIC deceptive patterns that statistical models miss.

Analyze the video for these specific flags:
1. **Piracy/Hack**: Does it offer downloads, cracks, mods, free money, or hacks? (e.g. "GTA 6 Download", "Unlimited Robux").
2. **Fake News/Scam**: Is it factually false or a known hoax? (e.g. "Avengers Doomsday Full Movie", "Celebrity Died" when alive).
3. **Duration Scam**: Does it claim "Full Movie" or "Full Episode" but is short (< 30 mins)?
4. **False Official**: Does it claim to be "Official Trailer" but is from a random channel?

Return JSON only:
{
    "is_clickbait": boolean,
    "confidence": 0.0 to 1.0,
    "reason": "Short reason",
    "flags": ["piracy", "fake_news", "duration_mismatch", "false_official"] (or empty list)
}
"""

# ============================================================
# HELPER FUNCTIONS (From V3)
# ============================================================
YOUTUBE_CATEGORIES = {"1": "Film & Animation", "2": "Autos & Vehicles", "10": "Music", "15": "Pets & Animals", "17": "Sports", "18": "Short Movies", "19": "Travel & Events", "20": "Gaming", "21": "Videoblogging", "22": "People & Blogs", "23": "Comedy", "24": "Entertainment", "25": "News & Politics", "26": "Howto & Style", "27": "Education", "28": "Science & Technology", "29": "Nonprofits & Activism", "30": "Movies", "31": "Anime/Animation", "32": "Action/Adventure", "33": "Classics", "34": "Comedy", "35": "Documentary", "36": "Drama", "37": "Family", "38": "Foreign", "39": "Horror", "40": "Sci-Fi/Fantasy", "41": "Thriller", "42": "Shorts", "43": "Shows", "44": "Trailers"}
CATEGORY_MAPPING = {"Film & Animation": "Movies_Bollywood_Clickbait_Queries", "Music": "Entertainment_Celebrity_Gossip_Clickbait_Queries", "Gaming": "Gaming_Clickbait_Queries", "Entertainment": "Entertainment_Celebrity_Gossip_Clickbait_Queries", "Comedy": "Entertainment_Celebrity_Gossip_Clickbait_Queries", "People & Blogs": "Entertainment_Celebrity_Gossip_Clickbait_Queries", "News & Politics": "News_Sensational_India_Clickbait_Queries", "Education": "Education_Exams_Clickbait_Queries", "Science & Technology": "Technology_Clickbait_Queries", "Howto & Style": "Technology_Clickbait_Queries", "Sports": "Sports_Cricket_Clickbait_Queries", "Autos & Vehicles": "Technology_Clickbait_Queries", "Travel & Events": "Entertainment_Celebrity_Gossip_Clickbait_Queries", "Pets & Animals": "Entertainment_Celebrity_Gossip_Clickbait_Queries", "Nonprofits & Activism": "News_Sensational_India_Clickbait_Queries", "Movies": "Movies_Bollywood_Clickbait_Queries"}
DEFAULT_CATEGORY = "Technology_Clickbait_Queries"

CLICKBAIT_KEYWORDS = ['shocking', 'exposed', 'truth', 'secret', 'viral', 'leaked', "you won't believe", 'must watch', 'watch till end', 'nobody tells', 'miracle', 'guaranteed', 'speechless', 'exclusive', 'breaking', 'urgent', 'warning', 'banned', 'deleted', 'hidden', 'revealed']
PIRACY_KEYWORDS = ['download', 'telegram', 'camrip', 'dvdrip', 'hdrip', 'torrent', 'leaked', 'bolly4u', 'filmyzilla', 'hdcam', 'pre-dvd', 'webrip', 'apk', 'mod apk', 'crack', 'patch', 'hack', 'cheat', 'glitch', 'unlimited', 'generator']
URGENCY_KEYWORDS = ['urgent', 'breaking', 'now', 'today', 'hurry', 'limited time', 'last chance', 'act now', 'immediately']
MONEY_KEYWORDS = ['free', 'earn', 'money', 'rich', 'millionaire', 'income', 'profit', 'cash', 'salary', 'lakhs', 'crores', 'rupees', '₹', 'bitcoin', 'crypto']
HEALTH_KEYWORDS = ['cure', 'miracle', 'weight loss', 'fat', 'diet', 'doctor', 'hospital', 'treatment', 'medicine', 'natural remedy', 'home remedy', 'disease']
CURIOSITY_KEYWORDS = ['secret', 'hidden', 'mystery', 'unknown', 'discover', 'truth', 'what happened', 'you didn\'t know', 'nobody knows', 'revealed']
FAKE_NEWS_KEYWORDS = ['exposed', 'scam', 'fraud', 'fake', 'hoax', 'lie', 'conspiracy', 'cover up', 'truth revealed']
EMOTIONAL_EMOJIS = ['😱', '🔥', '☠️', '💥', '🤯', '😶', '😭', '😡', '💀', '⚠️']

def parse_duration(duration_str):
    if not duration_str: return 0.0
    duration_str = duration_str.replace('PT', '')
    hours = 0; minutes = 0; seconds = 0
    if 'H' in duration_str: hours, duration_str = duration_str.split('H'); hours = int(hours)
    if 'M' in duration_str: minutes, duration_str = duration_str.split('M'); minutes = int(minutes)
    if 'S' in duration_str: seconds = int(duration_str.replace('S', ''))
    return hours * 60 + minutes + seconds / 60

def count_keywords(text, keywords):
    text_lower = str(text).lower()
    # Simple substring match (kept for compatibility with trained model features)
    return sum(1 for kw in keywords if kw in text_lower)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def simple_polarity(text):
    text = str(text).lower()
    positive = ['good', 'great', 'best', 'amazing', 'awesome', 'excellent', 'love', 'happy', 'perfect']
    negative = ['bad', 'worst', 'terrible', 'horrible', 'hate', 'ugly', 'awful', 'angry', 'scam', 'fake']
    pos = sum(1 for w in positive if w in text)
    neg = sum(1 for w in negative if w in text)
    total = pos + neg
    return (pos - neg) / total if total > 0 else 0.0

def simple_subjectivity(text):
    text = str(text).lower()
    sub = ['i think', 'believe', 'opinion', 'probably', 'maybe', 'seems', 'best', 'worst', 'amazing', 'terrible']
    count = sum(1 for w in sub if w in text)
    wc = len(text.split())
    return min(count / (wc / 10 + 1), 1.0) if wc > 0 else 0.0

def word_overlap(text1, text2):
    w1 = set(str(text1).lower().split())
    w2 = set(str(text2).lower().split())
    if not w1 or not w2: return 0.0
    return len(w1.intersection(w2)) / min(len(w1), len(w2))

def tfidf_similarity(text1, text2):
    w1 = set(str(text1).lower().split())
    w2 = set(str(text2).lower().split())
    if not w1 or not w2: return 0.0
    u = w1.union(w2)
    return len(w1.intersection(w2)) / len(u) if u else 0.0

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

def extract_features(title, description, category, duration_min, views, likes, thumbnail_text=""):
    data = {"title": [title], "description": [description], "thumbnail_text_cleaned": [thumbnail_text], "category": [category], "duration_min": [duration_min], "views": [views], "likes": [likes], "thumbnail_text_valid": [1 if thumbnail_text else 0]}
    df = pd.DataFrame(data)
    df["text"] = df["title"] + " " + df["description"] + " " + df["thumbnail_text_cleaned"]
    
    # Minimal features for model compatibility
    df["title_length"] = df["title"].str.len()
    df["desc_length"] = df["description"].str.len()
    df["title_word_count"] = df["title"].str.split().str.len().fillna(0)
    df["desc_word_count"] = df["description"].str.split().str.len().fillna(0)
    df["caps_ratio"] = df["title"].apply(lambda x: sum(1 for c in str(x) if c.isupper()) / max(len(str(x)), 1))
    df["title_caps_words"] = df["title"].apply(lambda x: sum(1 for w in str(x).split() if w.isupper() and len(w) > 1))
    df["question_count"] = df["title"].str.count(r"\?")
    df["exclam_count"] = df["title"].str.count(r"!")
    df["ellipsis_count"] = df["title"].str.count(r"\.\.\.")
    df["pipe_count"] = df["title"].str.count(r"\|")
    df["emoji_count"] = df["title"].apply(lambda x: sum(1 for c in str(x) if ord(c) > 127462))
    df["emotional_emoji_count"] = df["title"].apply(lambda x: sum(1 for e in EMOTIONAL_EMOJIS if e in str(x)))
    
    # Keywords
    df["clickbait_keywords"] = df["title"].apply(lambda x: count_keywords(x, CLICKBAIT_KEYWORDS))
    df["piracy_keywords"] = (df["title"].apply(lambda x: count_keywords(x, PIRACY_KEYWORDS)) + df["description"].apply(lambda x: count_keywords(x, PIRACY_KEYWORDS)))
    df["urgency_keywords"] = df["title"].apply(lambda x: count_keywords(x, URGENCY_KEYWORDS))
    df["money_keywords"] = df["title"].apply(lambda x: count_keywords(x, MONEY_KEYWORDS))
    df["health_keywords"] = df["title"].apply(lambda x: count_keywords(x, HEALTH_KEYWORDS))
    df["curiosity_keywords"] = df["title"].apply(lambda x: count_keywords(x, CURIOSITY_KEYWORDS))
    df["fake_news_keywords"] = df["title"].apply(lambda x: count_keywords(x, FAKE_NEWS_KEYWORDS))
    df["total_suspicious_keywords"] = (df["clickbait_keywords"] + df["urgency_keywords"] + df["money_keywords"] + df["health_keywords"] + df["curiosity_keywords"] + df["fake_news_keywords"])
    
    # Misc
    df["desc_is_empty"] = (df["desc_length"] < 20).astype(int)
    df["desc_hashtag_count"] = df["description"].str.count(r"#")
    df["desc_hashtag_ratio"] = df["desc_hashtag_count"] / (df["desc_word_count"] + 1)
    df["desc_has_links"] = df["description"].str.contains(r"http|https|www\.", regex=True).astype(int)
    
    # Regex Patterns
    df["has_full_movie_claim"] = df["title"].str.lower().str.contains(r"full movie|full hindi movie|full hd movie|complete movie", regex=True).astype(int)
    df["has_year_in_title"] = df["title"].str.contains(r"\b20[0-2][0-9]\b", regex=True).astype(int)
    df["has_hd_4k"] = df["title"].str.lower().str.contains(r"\bhd\b|\b4k\b|\b1080p\b", regex=True).astype(int)
    df["has_number_list"] = df["title"].str.contains(r"\b\d+\s*(?:things|ways|tips|tricks|reasons|steps|facts)\b", regex=True, case=False).astype(int)
    
    # Engagement
    df["likes_view_ratio"] = df["likes"] / (df["views"] + 1)
    df["likes_per_minute"] = df["likes"] / (df["duration_min"] + 0.1)
    df["views_per_minute"] = df["views"] / (df["duration_min"] + 0.1)
    df["log_views"] = np.log1p(df["views"])
    df["log_likes"] = np.log1p(df["likes"])
    df["log_duration"] = np.log1p(df["duration_min"])
    df["is_short_video"] = (df["duration_min"] < 1).astype(int)
    df["is_very_long"] = (df["duration_min"] > 60).astype(int)
    df["duration_mismatch"] = ((df["has_full_movie_claim"] == 1) & (df["duration_min"] < 60)).astype(int)
    df["engagement_score"] = (df["likes_view_ratio"] * 100 + np.log1p(df["views"]) / 10)
    df["low_engagement"] = ((df["likes_view_ratio"] < 0.001) & (df["views"] > 10000)).astype(int)
    
    # Sentiment
    df["title_polarity"] = df["title"].apply(simple_polarity)
    df["title_subjectivity"] = df["title"].apply(simple_subjectivity)
    df["desc_polarity"] = df["description"].apply(simple_polarity)
    df["desc_subjectivity"] = df["description"].apply(simple_subjectivity)
    df["polarity_diff"] = abs(df["title_polarity"] - df["desc_polarity"])
    df["subjectivity_diff"] = abs(df["title_subjectivity"] - df["desc_subjectivity"])
    df["title_very_positive"] = (df["title_polarity"] > 0.5).astype(int)
    df["title_very_negative"] = (df["title_polarity"] < -0.5).astype(int)
    df["title_very_subjective"] = (df["title_subjectivity"] > 0.6).astype(int)
    
    # Similarity
    df["title_desc_word_overlap"] = df.apply(lambda row: word_overlap(row["title"], row["description"]), axis=1)
    df["title_desc_tfidf_similarity"] = df.apply(lambda row: tfidf_similarity(row["title"], row["description"]), axis=1)
    df["low_title_desc_similarity"] = (df["title_desc_tfidf_similarity"] < 0.1).astype(int)
    
    df["text_clean"] = df["text"].apply(clean_text)
    return df

# ============================================================
# HYBRID LOGIC
# ============================================================

def analyze_semantic_flags(video_data):
    """
    Use Llama 3 to detect semantic flags (Expert Heuristics)
    that act as 'Confidence Boosters' for the LightGBM model.
    """
    user_content = f"""
    Title: {video_data['title']}
    Channel: {video_data['channel']}
    Duration: {video_data['duration_min']:.1f} minutes
    Stats: {video_data['views']} views, {video_data['likes']} likes
    Description: {video_data['description'][:500]}
    
    Check for flags.
    """
    try:
        completion = client.chat.completions.create(
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_content}],
            model="llama-3.3-70b",
            response_format={"type": "json_object"}
        )
        return json.loads(completion.choices[0].message.content)
    except:
        return {"is_clickbait": False, "confidence": 0, "flags": []}

def ensemble_predict(gbm_prob, llm_result):
    """
    Combine LightGBM probability with LLM Semantic Flags.
    """
    final_prob = gbm_prob
    reason = "Statistical Analysis"
    
    flags = llm_result.get("flags", [])
    
    if "piracy" in flags:
        final_prob = max(final_prob, 0.95)
        reason = "Detected Piracy/Mods/Hacks"
    elif "fake_news" in flags:
        final_prob = max(final_prob, 0.95)
        reason = "Detected Fake News/Scam"
    elif "duration_mismatch" in flags:
        final_prob = max(final_prob, 0.90)
        reason = "Duration does not match content claims"
    elif "false_official" in flags:
        final_prob = max(final_prob, 0.85)
        reason = "Unofficial channel claiming official content"
    
    # Soft boost from LLM confidence if generic clickbait
    if llm_result.get("is_clickbait") and not flags:
        llm_conf = llm_result.get("confidence", 0.5)
        # Average the probabilities if LLM is confident
        if llm_conf > 0.7:
             final_prob = (gbm_prob + llm_conf) / 2
             reason = "Combined AI consensus"

    return final_prob, reason, flags


# ============================================================
# API ROUTES
# ============================================================

@app.route("/")
def home():
    return "Clickbait API v5 (Hybrid Ensemble: LightGBM + Llama 3) Running"

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
        df = extract_features(video_data["title"], video_data["description"], video_data["category"], video_data["duration_min"], video_data["views"], video_data["likes"])
        X_text = tfidf.transform(df["text_clean"])
        X_num = scaler.transform(df[num_features].values)
        X_cat = cat_encoder.transform(df[["category"]])
        X_new = hstack([X_text, csr_matrix(X_num), csr_matrix(X_cat)])
        probs = model.predict_proba(X_new)[0]
        gbm_prob = float(probs[1])

        print(f"📊 LightGBM Prob: {gbm_prob:.4f} | Video: {video_data['title'][:40]}...")

        # 3. LLM Semantic Analysis (Expert Heuristics)
        llm_result = analyze_semantic_flags(video_data)
        print(f"🧠 LLM Flags: {llm_result.get('flags')} | Conf: {llm_result.get('confidence')}")

        # 4. Ensemble Combination
        final_prob, reason, flags = ensemble_predict(gbm_prob, llm_result)
        
        label = "Misleading" if final_prob > 0.5 else "Trustworthy"
        confidence = final_prob if label == "Misleading" else 1.0 - final_prob

        response = {
            "prediction": label,
            "confidence": float(confidence),
            "clickbait_probability": float(final_prob),
            "reason": reason,
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
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

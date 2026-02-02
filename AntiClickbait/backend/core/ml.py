import os
import joblib
import pandas as pd
import numpy as np
import re
from scipy.sparse import hstack, csr_matrix
from config import MODEL_DIR
from core.utils import (
    clean_text, count_keywords, simple_polarity, simple_subjectivity, 
    word_overlap, tfidf_similarity, EMOTIONAL_EMOJIS
)

# ============================================================
# KEYWORD LISTS (ML FEATURES)
# ============================================================
CLICKBAIT_KEYWORDS = ['shocking', 'exposed', 'truth', 'secret', 'viral', 'leaked', "you won't believe", 'must watch', 'watch till end', 'nobody tells', 'miracle', 'guaranteed', 'speechless', 'exclusive', 'breaking', 'urgent', 'warning', 'banned', 'deleted', 'hidden', 'revealed']
PIRACY_KEYWORDS = ['download', 'telegram', 'camrip', 'dvdrip', 'hdrip', 'torrent', 'leaked', 'bolly4u', 'filmyzilla', 'hdcam', 'pre-dvd', 'webrip', 'apk', 'mod apk', 'crack', 'patch', 'hack', 'cheat', 'glitch', 'unlimited', 'generator']
URGENCY_KEYWORDS = ['urgent', 'breaking', 'now', 'today', 'hurry', 'limited time', 'last chance', 'act now', 'immediately']
MONEY_KEYWORDS = ['free', 'earn', 'money', 'rich', 'millionaire', 'income', 'profit', 'cash', 'salary', 'lakhs', 'crores', 'rupees', '₹', 'bitcoin', 'crypto']
HEALTH_KEYWORDS = ['cure', 'miracle', 'weight loss', 'fat', 'diet', 'doctor', 'hospital', 'treatment', 'medicine', 'natural remedy', 'home remedy', 'disease']
CURIOSITY_KEYWORDS = ['secret', 'hidden', 'mystery', 'unknown', 'discover', 'truth', 'what happened', 'you didn\'t know', 'nobody knows', 'revealed']
FAKE_NEWS_KEYWORDS = ['exposed', 'scam', 'fraud', 'fake', 'hoax', 'lie', 'conspiracy', 'cover up', 'truth revealed']

# ============================================================
# MODEL LOADING
# ============================================================
try:
    print("⏳ Loading LightGBM Model...", flush=True)
    model = joblib.load(os.path.join(MODEL_DIR, "models", "clickbait_model_v2.joblib"))
    tfidf = joblib.load(os.path.join(MODEL_DIR, "models", "tfidf_vectorizer_v2.joblib"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "models", "scaler_v2.joblib"))
    cat_encoder = joblib.load(os.path.join(MODEL_DIR, "models", "cat_encoder_v2.joblib"))
    num_features_list = joblib.load(os.path.join(MODEL_DIR, "models", "num_features_v2.joblib"))
    print(f"✅ Loaded LightGBM model with {len(num_features_list)} features", flush=True)
except Exception as e:
    print(f"❌ Error loading ML models: {e}", flush=True)
    raise e

def predict_clickbait_prob(video_data):
    """
    Given a video dictionary, returns the float probability (0-1) of it being clickbait using LightGBM.
    """
    df = extract_features(
        video_data["title"], 
        video_data["description"], 
        video_data["category"], 
        video_data["duration_min"], 
        video_data["views"], 
        video_data["likes"]
    )
    
    # Vectorization & Transformation
    X_text = tfidf.transform(df["text_clean"])
    X_num = scaler.transform(df[num_features_list].values)
    X_cat = cat_encoder.transform(df[["category"]])
    
    # Combine
    X_new = hstack([X_text, csr_matrix(X_num), csr_matrix(X_cat)])
    
    # Predict
    probs = model.predict_proba(X_new)[0]
    return float(probs[1])

def extract_features(title, description, category, duration_min, views, likes, thumbnail_text=""):
    """
    Engineers the 55+ statistical features required by the LightGBM model.
    """
    data = {
        "title": [title], "description": [description], 
        "thumbnail_text_cleaned": [thumbnail_text], "category": [category], 
        "duration_min": [duration_min], "views": [views], "likes": [likes], 
        "thumbnail_text_valid": [1 if thumbnail_text else 0]
    }
    df = pd.DataFrame(data)
    df["text"] = df["title"] + " " + df["description"] + " " + df["thumbnail_text_cleaned"]
    
    # --- Semantic Features ---
    df["title_length"] = df["title"].str.len()
    df["desc_length"] = df["description"].str.len()
    df["title_word_count"] = df["title"].str.split().str.len().fillna(0)
    df["desc_word_count"] = df["description"].str.split().str.len().fillna(0)
    df["caps_ratio"] = df["title"].apply(lambda x: sum(1 for c in str(x) if c.isupper()) / max(len(str(x)), 1))
    df["title_caps_words"] = df["title"].apply(lambda x: sum(1 for w in str(x).split() if w.isupper() and len(w) > 1))
    
    # --- Punctuation ---
    df["question_count"] = df["title"].str.count(r"\?")
    df["exclam_count"] = df["title"].str.count(r"!")
    df["ellipsis_count"] = df["title"].str.count(r"\.\.\.")
    df["pipe_count"] = df["title"].str.count(r"\|")
    df["emoji_count"] = df["title"].apply(lambda x: sum(1 for c in str(x) if ord(c) > 127462))
    df["emotional_emoji_count"] = df["title"].apply(lambda x: sum(1 for e in EMOTIONAL_EMOJIS if e in str(x)))
    
    # --- Keywords Analysis ---
    df["clickbait_keywords"] = df["title"].apply(lambda x: count_keywords(x, CLICKBAIT_KEYWORDS))
    df["piracy_keywords"] = (df["title"].apply(lambda x: count_keywords(x, PIRACY_KEYWORDS)) + df["description"].apply(lambda x: count_keywords(x, PIRACY_KEYWORDS)))
    df["urgency_keywords"] = df["title"].apply(lambda x: count_keywords(x, URGENCY_KEYWORDS))
    df["money_keywords"] = df["title"].apply(lambda x: count_keywords(x, MONEY_KEYWORDS))
    df["health_keywords"] = df["title"].apply(lambda x: count_keywords(x, HEALTH_KEYWORDS))
    df["curiosity_keywords"] = df["title"].apply(lambda x: count_keywords(x, CURIOSITY_KEYWORDS))
    df["fake_news_keywords"] = df["title"].apply(lambda x: count_keywords(x, FAKE_NEWS_KEYWORDS))
    df["total_suspicious_keywords"] = (df["clickbait_keywords"] + df["urgency_keywords"] + df["money_keywords"] + df["health_keywords"] + df["curiosity_keywords"] + df["fake_news_keywords"])
    
    # --- Metadata & Formatting ---
    df["desc_is_empty"] = (df["desc_length"] < 20).astype(int)
    df["desc_hashtag_count"] = df["description"].str.count(r"#")
    df["desc_hashtag_ratio"] = df["desc_hashtag_count"] / (df["desc_word_count"] + 1)
    df["desc_has_links"] = df["description"].str.contains(r"http|https|www\.", regex=True).astype(int)
    
    # --- Regex Claims ---
    df["has_full_movie_claim"] = df["title"].str.lower().str.contains(r"full movie|full hindi movie|full hd movie|complete movie", regex=True).astype(int)
    df["has_year_in_title"] = df["title"].str.contains(r"\b20[0-2][0-9]\b", regex=True).astype(int)
    df["has_hd_4k"] = df["title"].str.lower().str.contains(r"\bhd\b|\b4k\b|\b1080p\b", regex=True).astype(int)
    df["has_number_list"] = df["title"].str.contains(r"\b\d+\s*(?:things|ways|tips|tricks|reasons|steps|facts)\b", regex=True, case=False).astype(int)
    
    # --- Engagement Stats ---
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
    
    # --- Advanced Sentiment ---
    df["title_polarity"] = df["title"].apply(simple_polarity)
    df["title_subjectivity"] = df["title"].apply(simple_subjectivity)
    df["desc_polarity"] = df["description"].apply(simple_polarity)
    df["desc_subjectivity"] = df["description"].apply(simple_subjectivity)
    df["polarity_diff"] = abs(df["title_polarity"] - df["desc_polarity"])
    df["subjectivity_diff"] = abs(df["title_subjectivity"] - df["desc_subjectivity"])
    df["title_very_positive"] = (df["title_polarity"] > 0.5).astype(int)
    df["title_very_negative"] = (df["title_polarity"] < -0.5).astype(int)
    df["title_very_subjective"] = (df["title_subjectivity"] > 0.6).astype(int)
    
    # --- Contextual Similarity ---
    df["title_desc_word_overlap"] = df.apply(lambda row: word_overlap(row["title"], row["description"]), axis=1)
    df["title_desc_tfidf_similarity"] = df.apply(lambda row: tfidf_similarity(row["title"], row["description"]), axis=1)
    df["low_title_desc_similarity"] = (df["title_desc_tfidf_similarity"] < 0.1).astype(int)
    
    df["text_clean"] = df["text"].apply(clean_text)
    return df

"""Debug script to test model predictions"""
import joblib
import pandas as pd
import numpy as np
import re
from scipy.sparse import hstack, csr_matrix
import warnings
warnings.filterwarnings('ignore')

# Load model components
print("Loading model components...")
model = joblib.load('BEST_FINAL_MODEL/clickbait_model.joblib')
tfidf = joblib.load('BEST_FINAL_MODEL/tfidf_vectorizer.joblib')
scaler = joblib.load('BEST_FINAL_MODEL/scaler.joblib')
cat_encoder = joblib.load('BEST_FINAL_MODEL/cat_encoder.joblib')
num_features = joblib.load('BEST_FINAL_MODEL/num_features.joblib')

print(f"Model type: {type(model).__name__}")
print(f"Num features expected: {len(num_features)}")
print(f"Features: {num_features}")

# Clickbait indicator keywords
CLICKBAIT_KEYWORDS = [
    'shocking', 'exposed', 'truth', 'secret', 'viral', 'leaked',
    "you won't believe", 'must watch', 'watch till end', 'nobody tells',
    'miracle', 'guaranteed', 'speechless', 'exclusive', 'breaking',
    'urgent', 'warning', 'banned', 'deleted', 'hidden', 'revealed'
]

PIRACY_KEYWORDS = [
    'download', 'telegram', 'camrip', 'dvdrip', 'hdrip', 'torrent',
    'leaked', 'bolly4u', 'filmyzilla', 'hdcam', 'pre-dvd', 'webrip'
]

EMOTIONAL_EMOJIS = ['😱', '🔥', '☠️', '💥', '🤯', '😶', '😭', '😡', '💀', '⚠️']

def count_keywords(text, keywords):
    text_lower = str(text).lower()
    return sum(1 for kw in keywords if kw in text_lower)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_features(title, description, category, duration_min, views, likes, thumbnail_text=""):
    data = {
        "title": [title],
        "description": [description],
        "thumbnail_text_cleaned": [thumbnail_text],
        "category": [category],
        "duration_min": [duration_min],
        "views": [views],
        "likes": [likes],
        "thumbnail_text_valid": [1 if thumbnail_text else 0]
    }
    df = pd.DataFrame(data)
    df["text"] = df["title"] + " " + df["description"] + " " + df["thumbnail_text_cleaned"]
    
    # Basic length features
    df["title_length"] = df["title"].str.len()
    df["desc_length"] = df["description"].str.len()
    df["title_word_count"] = df["title"].str.split().str.len().fillna(0)
    df["desc_word_count"] = df["description"].str.split().str.len().fillna(0)
    
    # Title style features
    df["caps_ratio"] = df["title"].apply(
        lambda x: sum(1 for c in str(x) if c.isupper()) / max(len(str(x)), 1)
    )
    df["title_caps_words"] = df["title"].apply(
        lambda x: sum(1 for w in str(x).split() if w.isupper() and len(w) > 1)
    )
    
    # Punctuation features
    df["question_count"] = df["title"].str.count(r"\?")
    df["exclam_count"] = df["title"].str.count(r"!")
    df["ellipsis_count"] = df["title"].str.count(r"\.\.\.")
    df["pipe_count"] = df["title"].str.count(r"\|")
    
    # Emoji features
    df["emoji_count"] = df["title"].apply(
        lambda x: sum(1 for c in str(x) if ord(c) > 127462)
    )
    df["emotional_emoji_count"] = df["title"].apply(
        lambda x: sum(1 for e in EMOTIONAL_EMOJIS if e in str(x))
    )
    
    # Clickbait keyword detection
    df["clickbait_keywords"] = df["title"].apply(lambda x: count_keywords(x, CLICKBAIT_KEYWORDS))
    df["piracy_keywords"] = (
        df["title"].apply(lambda x: count_keywords(x, PIRACY_KEYWORDS)) +
        df["description"].apply(lambda x: count_keywords(x, PIRACY_KEYWORDS))
    )
    
    # Description quality indicators
    df["desc_is_empty"] = (df["desc_length"] < 20).astype(int)
    df["desc_hashtag_count"] = df["description"].str.count(r"#")
    df["desc_hashtag_ratio"] = df["desc_hashtag_count"] / (df["desc_word_count"] + 1)
    df["desc_has_links"] = df["description"].str.contains(r"http|https|www\.", regex=True).astype(int)
    
    # Special patterns
    df["has_full_movie_claim"] = df["title"].str.lower().str.contains(
        r"full movie|full hindi movie|full hd movie|complete movie", regex=True
    ).astype(int)
    df["has_year_in_title"] = df["title"].str.contains(r"\b20[0-2][0-9]\b", regex=True).astype(int)
    df["has_hd_4k"] = df["title"].str.lower().str.contains(r"\bhd\b|\b4k\b|\b1080p\b", regex=True).astype(int)
    
    # Engagement ratios
    df["likes_view_ratio"] = df["likes"] / (df["views"] + 1)
    df["likes_per_minute"] = df["likes"] / (df["duration_min"] + 0.1)
    df["views_per_minute"] = df["views"] / (df["duration_min"] + 0.1)
    
    # Log-transformed features
    df["log_views"] = np.log1p(df["views"])
    df["log_likes"] = np.log1p(df["likes"])
    df["log_duration"] = np.log1p(df["duration_min"])
    
    # Duration-based features
    df["is_short_video"] = (df["duration_min"] < 1).astype(int)
    df["is_very_long"] = (df["duration_min"] > 60).astype(int)
    df["duration_mismatch"] = (
        (df["has_full_movie_claim"] == 1) & (df["duration_min"] < 60)
    ).astype(int)
    
    # Anomaly detection features
    df["engagement_score"] = (
        df["likes_view_ratio"] * 100 + 
        np.log1p(df["views"]) / 10
    )
    df["low_engagement"] = (
        (df["likes_view_ratio"] < 0.001) & (df["views"] > 10000)
    ).astype(int)
    
    # Clean text for vectorization
    df["text_clean"] = df["text"].apply(clean_text)
    
    return df


def predict(title, description, category, duration_min, views, likes, thumbnail_text=""):
    df = extract_features(title, description, category, duration_min, views, likes, thumbnail_text)
    
    # Transform features
    X_text_new = tfidf.transform(df["text_clean"])
    X_num_new = scaler.transform(df[num_features].values)
    X_cat_new = cat_encoder.transform(df[["category"]])
    
    # Combine all features
    X_new = hstack([X_text_new, csr_matrix(X_num_new), csr_matrix(X_cat_new)])
    
    # Predict
    pred = model.predict(X_new)[0]
    probs = model.predict_proba(X_new)[0]
    
    return pred, probs[0], probs[1]


print("\n" + "="*60)
print("TEST 1: Known Non-Clickbait from dataset (label=0)")
print("="*60)
pred, prob0, prob1 = predict(
    title="English Vinglish (HD) | Sridevi, Adil Hussain, Mehdi Nebbou",
    description="Watch the full movie of English Vinglish starring Sridevi.",
    category="Movies_Bollywood_Clickbait_Queries",
    duration_min=134.0,
    views=10104767,
    likes=107331
)
print(f"Prediction: {'Misleading' if pred == 1 else 'Trustworthy'}")
print(f"Prob(Trustworthy): {prob0:.4f}")
print(f"Prob(Misleading): {prob1:.4f}")


print("\n" + "="*60)
print("TEST 2: Obvious clickbait title")
print("="*60)
pred, prob0, prob1 = predict(
    title="SHOCKING! You WON'T BELIEVE what happened next!!! 😱🔥",
    description="Download from telegram link",
    category="Entertainment_Celebrity_Gossip_Clickbait_Queries",
    duration_min=2.0,
    views=500000,
    likes=1000
)
print(f"Prediction: {'Misleading' if pred == 1 else 'Trustworthy'}")
print(f"Prob(Trustworthy): {prob0:.4f}")
print(f"Prob(Misleading): {prob1:.4f}")


print("\n" + "="*60)
print("TEST 3: Normal educational video")
print("="*60)
pred, prob0, prob1 = predict(
    title="Python Tutorial for Beginners - Full Course 2024",
    description="Learn Python programming from scratch in this comprehensive tutorial.",
    category="Education_Exams_Clickbait_Queries",
    duration_min=180.0,
    views=2000000,
    likes=80000
)
print(f"Prediction: {'Misleading' if pred == 1 else 'Trustworthy'}")
print(f"Prob(Trustworthy): {prob0:.4f}")
print(f"Prob(Misleading): {prob1:.4f}")


print("\n" + "="*60)
print("TEST 4: Title only (like extension sends)")
print("="*60)
pred, prob0, prob1 = predict(
    title="Top 5 Best Powerbanks Under Rs1000",
    description="",
    category="Technology_Clickbait_Queries",
    duration_min=10.0,
    views=100000,
    likes=5000
)
print(f"Prediction: {'Misleading' if pred == 1 else 'Trustworthy'}")
print(f"Prob(Trustworthy): {prob0:.4f}")
print(f"Prob(Misleading): {prob1:.4f}")

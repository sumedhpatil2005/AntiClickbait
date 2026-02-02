"""Test model prediction using exact training pipeline on original dataset"""
import pandas as pd
import numpy as np
import re
import joblib
from scipy.sparse import hstack, csr_matrix
import warnings
warnings.filterwarnings('ignore')

# Load model components
model = joblib.load('BEST_FINAL_MODEL/clickbait_model.joblib')
tfidf = joblib.load('BEST_FINAL_MODEL/tfidf_vectorizer.joblib')
scaler = joblib.load('BEST_FINAL_MODEL/scaler.joblib')
cat_encoder = joblib.load('BEST_FINAL_MODEL/cat_encoder.joblib')
num_features = joblib.load('BEST_FINAL_MODEL/num_features.joblib')

print(f"Model: {type(model).__name__}")
print(f"TF-IDF vocabulary size: {len(tfidf.vocabulary_)}")
print(f"Scaler: mean shape = {scaler.mean_.shape}, scale shape = {scaler.scale_.shape}")
print(f"Num features: {len(num_features)}")

# Load and process dataset EXACTLY as notebook does
df = pd.read_csv('BEST_FINAL_MODEL/MASTER_DATASET.csv')
print(f"\nLoaded {len(df)} rows")

# Filter verified (same as notebook)
df = df[df["verified"] == 1].copy()

# Fill missing values (same as notebook)
text_cols = ["title", "description", "thumbnail_text_cleaned"]
for col in text_cols:
    df[col] = df[col].fillna("")

num_cols = ["duration_min", "views", "likes", "thumbnail_text_valid"]
for col in num_cols:
    df[col] = df[col].fillna(0)

# Create combined text (same as notebook)
df["text"] = df["title"] + " " + df["description"] + " " + df["thumbnail_text_cleaned"]

# Keywords
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

# Text features (same as notebook)
df["title_length"] = df["title"].str.len()
df["desc_length"] = df["description"].str.len()
df["title_word_count"] = df["title"].str.split().str.len().fillna(0)
df["desc_word_count"] = df["description"].str.split().str.len().fillna(0)

df["caps_ratio"] = df["title"].apply(
    lambda x: sum(1 for c in str(x) if c.isupper()) / max(len(str(x)), 1)
)
df["title_caps_words"] = df["title"].apply(
    lambda x: sum(1 for w in str(x).split() if w.isupper() and len(w) > 1)
)

df["question_count"] = df["title"].str.count(r"\?")
df["exclam_count"] = df["title"].str.count(r"!")
df["ellipsis_count"] = df["title"].str.count(r"\.\.\.")
df["pipe_count"] = df["title"].str.count(r"\|")

df["emoji_count"] = df["title"].apply(
    lambda x: sum(1 for c in str(x) if ord(c) > 127462)
)
df["emotional_emoji_count"] = df["title"].apply(
    lambda x: sum(1 for e in EMOTIONAL_EMOJIS if e in str(x))
)

def count_keywords(text, keywords):
    text_lower = str(text).lower()
    return sum(1 for kw in keywords if kw in text_lower)

df["clickbait_keywords"] = df["title"].apply(lambda x: count_keywords(x, CLICKBAIT_KEYWORDS))
df["piracy_keywords"] = (
    df["title"].apply(lambda x: count_keywords(x, PIRACY_KEYWORDS)) +
    df["description"].apply(lambda x: count_keywords(x, PIRACY_KEYWORDS))
)

df["desc_is_empty"] = (df["desc_length"] < 20).astype(int)
df["desc_hashtag_count"] = df["description"].str.count(r"#")
df["desc_hashtag_ratio"] = df["desc_hashtag_count"] / (df["desc_word_count"] + 1)
df["desc_has_links"] = df["description"].str.contains(r"http|https|www\.", regex=True).astype(int)

df["has_full_movie_claim"] = df["title"].str.lower().str.contains(
    r"full movie|full hindi movie|full hd movie|complete movie", regex=True
).astype(int)
df["has_year_in_title"] = df["title"].str.contains(r"\b20[0-2][0-9]\b", regex=True).astype(int)
df["has_hd_4k"] = df["title"].str.lower().str.contains(r"\bhd\b|\b4k\b|\b1080p\b", regex=True).astype(int)

# Engagement features (same as notebook)
df["likes_view_ratio"] = df["likes"] / (df["views"] + 1)
df["likes_per_minute"] = df["likes"] / (df["duration_min"] + 0.1)
df["views_per_minute"] = df["views"] / (df["duration_min"] + 0.1)

df["log_views"] = np.log1p(df["views"])
df["log_likes"] = np.log1p(df["likes"])
df["log_duration"] = np.log1p(df["duration_min"])

df["is_short_video"] = (df["duration_min"] < 1).astype(int)
df["is_very_long"] = (df["duration_min"] > 60).astype(int)
df["duration_mismatch"] = (
    (df["has_full_movie_claim"] == 1) & (df["duration_min"] < 60)
).astype(int)

df["engagement_score"] = (
    df["likes_view_ratio"] * 100 + 
    np.log1p(df["views"]) / 10
)
df["low_engagement"] = (
    (df["likes_view_ratio"] < 0.001) & (df["views"] > 10000)
).astype(int)

# Clean text for TF-IDF (same as notebook)
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df["text_clean"] = df["text"].apply(clean_text)

# Test on first 5 LABEL=0 samples
print("\n" + "="*60)
print("Testing on LABEL=0 samples from dataset")
print("="*60)

test_samples = df[df['label'] == 0].head(5)
for idx, row in test_samples.iterrows():
    sample_df = pd.DataFrame([row])
    
    # Transform using saved preprocessors
    X_text = tfidf.transform(sample_df["text_clean"])
    X_num = scaler.transform(sample_df[num_features].values)
    X_cat = cat_encoder.transform(sample_df[["category"]])
    
    X = hstack([X_text, csr_matrix(X_num), csr_matrix(X_cat)])
    
    pred = model.predict(X)[0]
    probs = model.predict_proba(X)[0]
    
    print(f"\nTitle: {row['title'][:50]}...")
    print(f"Actual label: {row['label']} | Predicted: {pred}")
    print(f"Prob[0]: {probs[0]:.4f}, Prob[1]: {probs[1]:.4f}")

# Test on first 5 LABEL=1 samples
print("\n" + "="*60)
print("Testing on LABEL=1 samples from dataset")
print("="*60)

test_samples = df[df['label'] == 1].head(5)
for idx, row in test_samples.iterrows():
    sample_df = pd.DataFrame([row])
    
    X_text = tfidf.transform(sample_df["text_clean"])
    X_num = scaler.transform(sample_df[num_features].values)
    X_cat = cat_encoder.transform(sample_df[["category"]])
    
    X = hstack([X_text, csr_matrix(X_num), csr_matrix(X_cat)])
    
    pred = model.predict(X)[0]
    probs = model.predict_proba(X)[0]
    
    print(f"\nTitle: {row['title'][:50]}...")
    print(f"Actual label: {row['label']} | Predicted: {pred}")
    print(f"Prob[0]: {probs[0]:.4f}, Prob[1]: {probs[1]:.4f}")

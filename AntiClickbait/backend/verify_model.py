"""Quick verification test for the new model"""
import joblib
import pandas as pd
import numpy as np
import re
from scipy.sparse import hstack, csr_matrix
import warnings
warnings.filterwarnings('ignore')

print("Loading model...")
model = joblib.load('BEST_FINAL_MODEL/clickbait_model.joblib')
tfidf = joblib.load('BEST_FINAL_MODEL/tfidf_vectorizer.joblib')
scaler = joblib.load('BEST_FINAL_MODEL/scaler.joblib')
cat_encoder = joblib.load('BEST_FINAL_MODEL/cat_encoder.joblib')
num_features = joblib.load('BEST_FINAL_MODEL/num_features.joblib')

# Load and prepare dataset
df = pd.read_csv('BEST_FINAL_MODEL/MASTER_DATASET.csv')
df = df[df['verified'] == 1].copy()

for col in ['title', 'description', 'thumbnail_text_cleaned']:
    df[col] = df[col].fillna('')
for col in ['duration_min', 'views', 'likes', 'thumbnail_text_valid']:
    df[col] = df[col].fillna(0)

df['text'] = df['title'] + ' ' + df['description'] + ' ' + df['thumbnail_text_cleaned']

# Keywords and helpers
CLICKBAIT_KEYWORDS = ['shocking', 'exposed', 'truth', 'secret', 'viral', 'leaked', "you won't believe", 'must watch']
PIRACY_KEYWORDS = ['download', 'telegram', 'camrip', 'dvdrip', 'torrent', 'leaked']
EMOTIONAL_EMOJIS = ['😱', '🔥', '☠️', '💥', '🤯']

def count_keywords(text, keywords):
    return sum(1 for kw in keywords if kw in str(text).lower())

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

# Feature engineering
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
df["clickbait_keywords"] = df["title"].apply(lambda x: count_keywords(x, CLICKBAIT_KEYWORDS))
df["piracy_keywords"] = df["title"].apply(lambda x: count_keywords(x, PIRACY_KEYWORDS)) + df["description"].apply(lambda x: count_keywords(x, PIRACY_KEYWORDS))
df["desc_is_empty"] = (df["desc_length"] < 20).astype(int)
df["desc_hashtag_count"] = df["description"].str.count(r"#")
df["desc_hashtag_ratio"] = df["desc_hashtag_count"] / (df["desc_word_count"] + 1)
df["desc_has_links"] = df["description"].str.contains(r"http|https|www\.", regex=True).astype(int)
df["has_full_movie_claim"] = df["title"].str.lower().str.contains(r"full movie|full hindi movie", regex=True).astype(int)
df["has_year_in_title"] = df["title"].str.contains(r"\b20[0-2][0-9]\b", regex=True).astype(int)
df["has_hd_4k"] = df["title"].str.lower().str.contains(r"\bhd\b|\b4k\b", regex=True).astype(int)
df["likes_view_ratio"] = df["likes"] / (df["views"] + 1)
df["likes_per_minute"] = df["likes"] / (df["duration_min"] + 0.1)
df["views_per_minute"] = df["views"] / (df["duration_min"] + 0.1)
df["log_views"] = np.log1p(df["views"])
df["log_likes"] = np.log1p(df["likes"])
df["log_duration"] = np.log1p(df["duration_min"])
df["is_short_video"] = (df["duration_min"] < 1).astype(int)
df["is_very_long"] = (df["duration_min"] > 60).astype(int)
df["duration_mismatch"] = ((df["has_full_movie_claim"] == 1) & (df["duration_min"] < 60)).astype(int)
df["engagement_score"] = df["likes_view_ratio"] * 100 + np.log1p(df["views"]) / 10
df["low_engagement"] = ((df["likes_view_ratio"] < 0.001) & (df["views"] > 10000)).astype(int)
df["text_clean"] = df["text"].apply(clean_text)

print("\n" + "="*50)
print("TESTING NON-CLICKBAIT SAMPLES (label=0)")
print("="*50)
correct_0 = 0
for idx, row in df[df['label'] == 0].head(5).iterrows():
    sample_df = pd.DataFrame([row])
    X_text = tfidf.transform(sample_df["text_clean"])
    X_num = scaler.transform(sample_df[num_features].values)
    X_cat = cat_encoder.transform(sample_df[["category"]])
    X = hstack([X_text, csr_matrix(X_num), csr_matrix(X_cat)])
    pred = model.predict(X)[0]
    probs = model.predict_proba(X)[0]
    status = "OK" if pred == 0 else "WRONG"
    if pred == 0:
        correct_0 += 1
    print(f"{status}: Actual=0, Predicted={pred}, P(0)={probs[0]:.3f}, P(1)={probs[1]:.3f}")

print("\n" + "="*50)
print("TESTING CLICKBAIT SAMPLES (label=1)")  
print("="*50)
correct_1 = 0
for idx, row in df[df['label'] == 1].head(5).iterrows():
    sample_df = pd.DataFrame([row])
    X_text = tfidf.transform(sample_df["text_clean"])
    X_num = scaler.transform(sample_df[num_features].values)
    X_cat = cat_encoder.transform(sample_df[["category"]])
    X = hstack([X_text, csr_matrix(X_num), csr_matrix(X_cat)])
    pred = model.predict(X)[0]
    probs = model.predict_proba(X)[0]
    status = "OK" if pred == 1 else "WRONG"
    if pred == 1:
        correct_1 += 1
    print(f"{status}: Actual=1, Predicted={pred}, P(0)={probs[0]:.3f}, P(1)={probs[1]:.3f}")

print("\n" + "="*50)
print(f"RESULTS: correct 0s = {correct_0}/5, correct 1s = {correct_1}/5")
print(f"TOTAL: {correct_0 + correct_1}/10")
print("="*50)

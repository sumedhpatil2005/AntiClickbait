"""Quick test of v2 model prediction"""
import joblib
import pandas as pd
import numpy as np
import re
from scipy.sparse import hstack, csr_matrix
import warnings
warnings.filterwarnings('ignore')

# Load v2 model
print("Loading v2 model...")
model = joblib.load('BEST_FINAL_MODEL/clickbait_model_v2.joblib')
tfidf = joblib.load('BEST_FINAL_MODEL/tfidf_vectorizer_v2.joblib')
scaler = joblib.load('BEST_FINAL_MODEL/scaler_v2.joblib')
cat_encoder = joblib.load('BEST_FINAL_MODEL/cat_encoder_v2.joblib')
num_features = joblib.load('BEST_FINAL_MODEL/num_features_v2.joblib')

# Import extract_features from app
from app import extract_features

print(f"Model: {len(num_features)} numerical features")

# Test 1: Normal video title
print("\n" + "="*50)
print("TEST 1: Normal video")
df = extract_features(
    title="Python Tutorial for Beginners - Full Course",
    description="Learn Python programming",
    category="Education_Exams_Clickbait_Queries",
    duration_min=120.0, views=500000, likes=25000
)
X_text = tfidf.transform(df["text_clean"])
X_num = scaler.transform(df[num_features].values)
X_cat = cat_encoder.transform(df[["category"]])
X = hstack([X_text, csr_matrix(X_num), csr_matrix(X_cat)])
pred = model.predict(X)[0]
probs = model.predict_proba(X)[0]
print(f"Result: {'CLICKBAIT' if pred==1 else 'NOT CLICKBAIT'}")
print(f"Prob: {probs[1]*100:.1f}% clickbait")

# Test 2: Clickbait title
print("\n" + "="*50)
print("TEST 2: Clickbait title")
df = extract_features(
    title="SHOCKING! You Won't BELIEVE What Happened!! 😱🔥",
    description="Click now limited time",
    category="Entertainment_Celebrity_Gossip_Clickbait_Queries",
    duration_min=3.0, views=100000, likes=500
)
X_text = tfidf.transform(df["text_clean"])
X_num = scaler.transform(df[num_features].values)
X_cat = cat_encoder.transform(df[["category"]])
X = hstack([X_text, csr_matrix(X_num), csr_matrix(X_cat)])
pred = model.predict(X)[0]
probs = model.predict_proba(X)[0]
print(f"Result: {'CLICKBAIT' if pred==1 else 'NOT CLICKBAIT'}")
print(f"Prob: {probs[1]*100:.1f}% clickbait")

print("\n✅ v2 model test complete!")

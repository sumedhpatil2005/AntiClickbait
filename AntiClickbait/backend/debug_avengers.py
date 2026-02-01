"""Debug script for Avengers video prediction"""
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

from app import extract_features

# Avengers Video Data from logs
title = "Avengers: Doomsday (Full Movie) | Marvel Studios"
description = "Avengers Doomsday Full Movie Watch Online Free... download link..."
category = "Movies_Bollywood_Clickbait_Queries"
duration_min = 133.4
views = 503976
likes = 2003
thumbnail_text = ""

print("\n" + "="*50)
print("DEBUG: Feature Extraction")
print("="*50)

# Extract features
df = extract_features(title, description, category, duration_min, views, likes, thumbnail_text)

# Print key features
print(f"Title: {title}")
print(f"Category: {category}")
print(f"Has Full Movie Claim: {df['has_full_movie_claim'].values[0]}")
print(f"Piracy Keywords: {df['piracy_keywords'].values[0]}")
print(f"Clickbait Keywords: {df['clickbait_keywords'].values[0]}")
print(f"Views: {views}, Likes: {likes}")
print(f"Likes/View Ratio: {df['likes_view_ratio'].values[0]:.4f}")
print(f"Engagement Score: {df['engagement_score'].values[0]:.4f}")

# Transform
X_text = tfidf.transform(df["text_clean"])
X_num = scaler.transform(df[num_features].values)
X_cat = cat_encoder.transform(df[["category"]])
X = hstack([X_text, csr_matrix(X_num), csr_matrix(X_cat)])

# Predict
pred = model.predict(X)[0]
probs = model.predict_proba(X)[0]

print("\n" + "="*50)
print("PREDICTION RESULTS")
print("="*50)
print(f"Prediction: {pred} ({'Misleading' if pred==1 else 'Trustworthy'})")
print(f"Confidence (Clickbait as 1): {probs[1]:.4f}")
print(f"Confidence (Trustworthy as 0): {probs[0]:.4f}")

# Feature Contribution Analysis (roughly)
print("\nChecking Scaled Numerical Features (Top contributors usually):")
for i, feat in enumerate(num_features):
    val = df[feat].values[0]
    scaled_val = X_num[0][i]
    if abs(scaled_val) > 1.0: # Print only significantly scaled features
        print(f"  {feat}: Raw={val:.4f}, Scaled={scaled_val:.4f}")

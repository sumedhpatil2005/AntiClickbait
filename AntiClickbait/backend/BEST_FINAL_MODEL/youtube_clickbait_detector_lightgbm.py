# -*- coding: utf-8 -*-
"""
YouTube Clickbait Detector - LightGBM Model
Local Training Script (Fixed Version)

This script trains a LightGBM model for YouTube clickbait detection.
Run from the BEST_FINAL_MODEL directory:
    python youtube_clickbait_detector_lightgbm.py

Features:
- Advanced text preprocessing with TF-IDF vectorization
- Comprehensive feature engineering (36 features)
- LightGBM classifier with optimized hyperparameters
- Model persistence for deployment
- Post-training verification
"""

import pandas as pd
import numpy as np
import re
import os
import warnings
import joblib
from typing import Tuple, Dict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, precision_recall_curve
from scipy.sparse import hstack, csr_matrix
from lightgbm import LGBMClassifier

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
DATASET_PATH = "MASTER_DATASET.csv"
OUTPUT_DIR = "."
MAX_TFIDF_FEATURES = 5000
RANDOM_STATE = 42

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


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def count_keywords(text, keywords):
    """Count occurrences of keywords in text."""
    text_lower = str(text).lower()
    return sum(1 for kw in keywords if kw in text_lower)


def clean_text(text):
    """Clean text for TF-IDF vectorization."""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ============================================================
# DATA LOADING
# ============================================================

def load_and_prepare_data(filepath: str) -> pd.DataFrame:
    """Load and prepare the dataset with initial cleaning."""
    print("=" * 60)
    print("📂 LOADING DATA")
    print("=" * 60)

    df = pd.read_csv(filepath)
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Keep only verified rows
    if "verified" in df.columns:
        df = df[df["verified"] == 1].copy()
        print(f"After filtering verified: {len(df)} rows")

    # Fill missing values
    text_cols = ["title", "description", "thumbnail_text_cleaned"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("")

    num_cols = ["duration_min", "views", "likes", "thumbnail_text_valid"]
    for col in num_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Create combined text field
    df["text"] = df["title"] + " " + df["description"] + " " + df["thumbnail_text_cleaned"]

    print(f"\n📊 Class distribution:")
    print(df["label"].value_counts())
    print(f"Clickbait ratio: {df['label'].mean()*100:.1f}%")

    return df


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def extract_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract comprehensive text-based features."""
    print("\n" + "=" * 60)
    print("🔤 FEATURE ENGINEERING - TEXT FEATURES")
    print("=" * 60)

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

    print("✅ Text features extracted successfully!")
    return df


def extract_engagement_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract engagement and metadata features."""
    print("\n" + "=" * 60)
    print("📈 FEATURE ENGINEERING - ENGAGEMENT FEATURES")
    print("=" * 60)

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

    # Low engagement flag
    df["low_engagement"] = (
        (df["likes_view_ratio"] < 0.001) & (df["views"] > 10000)
    ).astype(int)

    print("✅ Engagement features extracted successfully!")
    return df


# ============================================================
# BUILD FEATURE MATRIX
# ============================================================

def build_feature_matrix(df: pd.DataFrame, max_features: int = 5000) -> Tuple:
    """Build the complete feature matrix."""
    print("\n" + "=" * 60)
    print("🔨 BUILDING FEATURE MATRIX")
    print("=" * 60)

    # Clean text for vectorization
    df["text_clean"] = df["text"].apply(clean_text)

    # TF-IDF Vectorization
    tfidf = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.95,
        sublinear_tf=True
    )
    X_text = tfidf.fit_transform(df["text_clean"])
    print(f"TF-IDF features: {X_text.shape[1]}")

    # Category encoding
    cat_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_cat = cat_encoder.fit_transform(df[["category"]])
    print(f"Categories: {df['category'].nunique()}")

    # Numerical features
    num_features = [
        "duration_min", "views", "likes", "thumbnail_text_valid",
        "title_length", "desc_length", "title_word_count", "desc_word_count",
        "caps_ratio", "title_caps_words",
        "question_count", "exclam_count", "ellipsis_count", "pipe_count",
        "emoji_count", "emotional_emoji_count",
        "clickbait_keywords", "piracy_keywords",
        "desc_is_empty", "desc_hashtag_count", "desc_hashtag_ratio", "desc_has_links",
        "has_full_movie_claim", "has_year_in_title", "has_hd_4k",
        "likes_view_ratio", "likes_per_minute", "views_per_minute",
        "log_views", "log_likes", "log_duration",
        "is_short_video", "is_very_long", "duration_mismatch",
        "engagement_score", "low_engagement"
    ]

    X_num = df[num_features].values
    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(X_num)

    # Combine all features
    X = hstack([X_text, csr_matrix(X_num_scaled), csr_matrix(X_cat)])
    y = df["label"].values

    print(f"\n📐 Final feature matrix shape: {X.shape}")
    print(f"  - Text features: {X_text.shape[1]}")
    print(f"  - Numerical features: {len(num_features)}")
    print(f"  - Category features: {X_cat.shape[1]}")

    return X, y, tfidf, scaler, cat_encoder, num_features, df


# ============================================================
# MODEL TRAINING
# ============================================================

def train_lightgbm(X, y) -> Dict:
    """Train LightGBM model."""
    print("\n" + "=" * 60)
    print("🏆 TRAINING LIGHTGBM MODEL")
    print("=" * 60)

    # Split data: 70% train, 15% validation, 15% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=RANDOM_STATE, stratify=y_temp
    )

    print(f"📊 Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

    # Train LightGBM
    lgb_model = LGBMClassifier(
        n_estimators=500,
        max_depth=10,
        learning_rate=0.1,
        num_leaves=31,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1
    )

    print("\n🔄 Training in progress...")
    lgb_model.fit(X_train, y_train)

    # Validation metrics
    y_pred_val = lgb_model.predict(X_val)
    y_prob_val = lgb_model.predict_proba(X_val)[:, 1]
    val_f1 = f1_score(y_val, y_pred_val)
    val_auc = roc_auc_score(y_val, y_prob_val)

    print(f"\n✅ Validation Results:")
    print(f"   F1 Score: {val_f1:.4f}")
    print(f"   ROC-AUC: {val_auc:.4f}")

    return {
        "model": lgb_model,
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "val_f1": val_f1,
        "val_auc": val_auc
    }


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(model, X_test, y_test):
    """Evaluate model on test set."""
    print("\n" + "=" * 60)
    print("📋 FINAL EVALUATION ON TEST SET")
    print("=" * 60)

    y_pred_test = model.predict(X_test)
    y_prob_test = model.predict_proba(X_test)[:, 1]

    print("\n🔲 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_test))

    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred_test, target_names=["Non-Clickbait", "Clickbait"]))

    test_auc = roc_auc_score(y_test, y_prob_test)
    test_f1 = f1_score(y_test, y_pred_test)
    print(f"🎯 Test ROC-AUC: {test_auc:.4f}")
    print(f"🎯 Test F1 Score: {test_f1:.4f}")
    
    return test_f1, test_auc


def optimize_threshold(model, X_test, y_test):
    """Find optimal classification threshold."""
    print("\n" + "=" * 60)
    print("🎚️ THRESHOLD OPTIMIZATION")
    print("=" * 60)

    y_prob = model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

    # Calculate F1 for each threshold
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5

    print(f"Default threshold (0.5) F1: {f1_score(y_test, (y_prob > 0.5).astype(int)):.4f}")
    print(f"Optimal threshold ({optimal_threshold:.3f}) F1: {f1_scores[optimal_idx]:.4f}")

    return optimal_threshold


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(model, tfidf, scaler, cat_encoder, num_features, output_dir="."):
    """Save model and preprocessors for deployment."""
    print("\n" + "=" * 60)
    print("💾 SAVING MODEL")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)
    
    joblib.dump(model, f"{output_dir}/clickbait_model.joblib")
    joblib.dump(tfidf, f"{output_dir}/tfidf_vectorizer.joblib")
    joblib.dump(scaler, f"{output_dir}/scaler.joblib")
    joblib.dump(cat_encoder, f"{output_dir}/cat_encoder.joblib")
    joblib.dump(num_features, f"{output_dir}/num_features.joblib")

    print(f"✅ Model saved to: {output_dir}/clickbait_model.joblib")
    print("✅ All preprocessors saved successfully!")


# ============================================================
# VERIFICATION
# ============================================================

def verify_model(df, tfidf, scaler, cat_encoder, num_features, model):
    """Verify model predictions on known samples."""
    print("\n" + "=" * 60)
    print("🔍 POST-TRAINING VERIFICATION")
    print("=" * 60)
    
    # Test on label=0 samples
    print("\n📗 Testing on NON-CLICKBAIT samples (label=0):")
    test_0 = df[df['label'] == 0].head(5)
    correct_0 = 0
    for idx, row in test_0.iterrows():
        sample_df = pd.DataFrame([row])
        X_text = tfidf.transform(sample_df["text_clean"])
        X_num = scaler.transform(sample_df[num_features].values)
        X_cat = cat_encoder.transform(sample_df[["category"]])
        X = hstack([X_text, csr_matrix(X_num), csr_matrix(X_cat)])
        
        pred = model.predict(X)[0]
        probs = model.predict_proba(X)[0]
        
        status = "✅" if pred == 0 else "❌"
        if pred == 0:
            correct_0 += 1
        print(f"  {status} Pred={pred}, Prob[0]={probs[0]:.3f}, Prob[1]={probs[1]:.3f}")
    
    # Test on label=1 samples
    print("\n📕 Testing on CLICKBAIT samples (label=1):")
    test_1 = df[df['label'] == 1].head(5)
    correct_1 = 0
    for idx, row in test_1.iterrows():
        sample_df = pd.DataFrame([row])
        X_text = tfidf.transform(sample_df["text_clean"])
        X_num = scaler.transform(sample_df[num_features].values)
        X_cat = cat_encoder.transform(sample_df[["category"]])
        X = hstack([X_text, csr_matrix(X_num), csr_matrix(X_cat)])
        
        pred = model.predict(X)[0]
        probs = model.predict_proba(X)[0]
        
        status = "✅" if pred == 1 else "❌"
        if pred == 1:
            correct_1 += 1
        print(f"  {status} Pred={pred}, Prob[0]={probs[0]:.3f}, Prob[1]={probs[1]:.3f}")
    
    total_correct = correct_0 + correct_1
    print(f"\n🎯 Verification Accuracy: {total_correct}/10 ({total_correct*10}%)")
    
    if total_correct >= 7:
        print("✅ Model verification PASSED!")
        return True
    else:
        print("❌ Model verification FAILED - model may not be working correctly")
        return False


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "=" * 60)
    print("🎯 YOUTUBE CLICKBAIT DETECTOR - TRAINING")
    print("=" * 60)
    
    # Check dataset exists
    if not os.path.exists(DATASET_PATH):
        print(f"❌ ERROR: Dataset not found at: {DATASET_PATH}")
        print("Please ensure MASTER_DATASET.csv is in the current directory.")
        return
    
    # Load and prepare data
    df = load_and_prepare_data(DATASET_PATH)
    
    # Feature engineering
    df = extract_text_features(df)
    df = extract_engagement_features(df)
    
    # Build feature matrix
    X, y, tfidf, scaler, cat_encoder, num_features, df = build_feature_matrix(df, MAX_TFIDF_FEATURES)
    
    # Train model
    results = train_lightgbm(X, y)
    model = results["model"]
    
    # Evaluate
    evaluate_model(model, results["X_test"], results["y_test"])
    optimal_threshold = optimize_threshold(model, results["X_test"], results["y_test"])
    
    # Save model
    save_model(model, tfidf, scaler, cat_encoder, num_features, OUTPUT_DIR)
    
    # Verify model works correctly
    verified = verify_model(df, tfidf, scaler, cat_encoder, num_features, model)
    
    # Summary
    print("\n" + "=" * 60)
    print("🎉 TRAINING COMPLETE!")
    print("=" * 60)
    print(f"🏆 Model: LightGBM")
    print(f"📊 Validation F1: {results['val_f1']:.4f}")
    print(f"📊 Validation AUC: {results['val_auc']:.4f}")
    print(f"🎚️ Optimal Threshold: {optimal_threshold:.3f}")
    print(f"🔍 Verification: {'PASSED' if verified else 'FAILED'}")
    print("\n✅ Model files saved and ready for deployment!")
    print("\nNext steps:")
    print("  1. cd to backend folder")
    print("  2. Run: python app.py")
    print("  3. Reload Chrome extension")
    print("  4. Test on YouTube videos!")


if __name__ == "__main__":
    main()
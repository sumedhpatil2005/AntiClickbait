# BEST_FINAL_MODEL - Directory Structure

## 📁 Directory Organization

```
BEST_FINAL_MODEL/
├── MASTER_DATASET.csv              # Main dataset (3,813 videos)
│
├── models/                          # Trained model files
│   ├── clickbait_model_v2.joblib   # LightGBM model
│   ├── tfidf_vectorizer_v2.joblib  # TF-IDF vectorizer
│   ├── scaler_v2.joblib            # Standard scaler
│   ├── cat_encoder_v2.joblib       # Category encoder
│   └── num_features_v2.joblib      # Numerical features config
│
├── scripts/                         # Production scripts
│   ├── youtube_clickbait_detector_lightgbm.py  # Main detection script
│   └── extract_transcripts.py      # Transcript extraction (archive)
│
├── transcripts/                     # Transcript extraction files
│   ├── README_TRANSCRIPTS.md       # Transcript extraction guide
│   ├── requirements_transcripts.txt # Dependencies
│   └── transcript_checkpoint.csv   # Extraction checkpoint (archive)
│
├── tests/                          # Test and debug scripts
│   ├── test_adaptive.py           # Adaptive sampling test
│   ├── test_limits.py            # Limits testing
│   ├── test_transcript.py        # Transcript testing
│   └── debug_api.py              # API debugging
│
├── archive/                        # Old/unused files
│
├── Youtube_ClickBait_Detector_LightGBM.ipynb     # Original training notebook
└── Youtube_ClickBait_Detector_LightGBM_v2.ipynb  # Updated training notebook
```

## 🎯 Quick Access

### To run the clickbait detector:
```bash
python scripts/youtube_clickbait_detector_lightgbm.py
```

### To train new models:
Open `Youtube_ClickBait_Detector_LightGBM_v2.ipynb` in Jupyter/Colab

### For transcript extraction:
See `transcripts/README_TRANSCRIPTS.md` for full guide

## 📊 Dataset
- **Location**: `MASTER_DATASET.csv`
- **Videos**: 3,813
- **Features**: video_id, title, description, duration_min, views, likes, thumbnail info, label, etc.

## 🤖 Models
All trained model artifacts are in `models/` directory. Load them using:
```python
import joblib
model = joblib.load('models/clickbait_model_v2.joblib')
```

## 🔧 Development
- Test scripts: `tests/`
- Production scripts: `scripts/`
- Archived/old files: `archive/`

# AntiClickbait (2026 Edition) 🛡️

A powerful AI-powered browser extension that detects misleading YouTube clickbait handling real-world complexity (Piracy, Fake News, Duration Scams) using a **Hybrid Ensemble Architecture**.

![System Workflow](architecture_diagram.html)

## 🚀 Key Features

*   **Hybrid Intelligence**: Combines **LightGBM** (Statistical Model) for speed with **Llama 3** (LLM) for semantic reasoning.
*   **Deep Verification**: Fetches video transcripts to verify if the content actually matches the title.
*   **Piracy/Scam Shield**: Detects "Full Movie" fakes, unofficial cracks, and scams using strict channel verification.
*   **Real-time Logic**:
    *   **The "Glance"**: Instantly flags CAPS, emojis, and spammy stats.
    *   **The "Watch"**: Reads the video for you to find the "Key Moment".
    *   **The "Veto"**: Prevents false positives by verifying trusted channels.

## 🛠️ Installation

### 1. Backend Setup (Flask API)
The brains of the operation.

```bash
cd backend
pip install -r requirements.txt
```

**Configuration:**
1.  Duplicate `backend/api_config.example.py` and rename it to `backend/api_config.py`.
2.  Add your API keys (YouTube Data API, Cerebras/Llama, TranscriptAPI).

**Run:**
```bash
python app.py
```

### 2. Extension Setup (Chrome/Brave)
The eyes of the operation.

1.  Open Chrome and go to `chrome://extensions`.
2.  Enable **Developer Mode** (top right).
3.  Click **Load Unpacked**.
4.  Select the `extension` folder from this repository.
5.  Pin the extension and browse YouTube!

## 🧠 Architecture

See `architecture_diagram.html` (view in browser) or `project_workflow.md` for a detailed breakdown of the decision logic.

## 📂 Project Structure

*   `backend/` - Flask Server + ML Models + Logic
*   `extension/` - Javascript Chrome Extension
*   `notebooks/` - Research & Dataset generation
*   `BEST_FINAL_MODEL/` - Pre-trained LightGBM artifacts

## 📜 License
MIT

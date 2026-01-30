from flask import Flask, request, jsonify
import joblib
import os
from flask_cors import CORS


app = Flask(__name__)
CORS(app)
# Load model and vectorizer
MODEL_PATH = os.path.join("models", "clickbait_model.pkl")
VECTORIZER_PATH = os.path.join("models", "vectorizer.pkl")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


@app.route("/")
def home():
    return "Clickbait Detection API Running"


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        title = data.get("title", "")
        description = data.get("description", "")
        thumbnail_text = data.get("thumbnail_text", "")

        combined_text = title + " " + description + " " + thumbnail_text

        text_vec = vectorizer.transform([combined_text])
        pred = model.predict(text_vec)[0]
        prob = model.predict_proba(text_vec)[0][1]

        label = "Misleading" if pred == 1 else "Trustworthy"

        return jsonify({
            "prediction": label,
            "confidence": float(prob)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)

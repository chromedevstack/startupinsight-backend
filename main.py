from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup
from transformers import pipeline, T5ForConditionalGeneration, T5Tokenizer
import torch
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load models only once
print("Loading AI models...")
qa_pipeline = pipeline("question-answering")
model_name = "google/flan-t5-small"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)
print("Models loaded successfully.")

# Scrape text from a URL
def scrape_text_from_url(url):
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        text = ' '.join(p.get_text() for p in soup.find_all('p'))
        return text.strip()
    except Exception as e:
        print(f"Scraping error: {e}")
        return None

# --- Routes ---
@app.route("/")
def home():
    return "StartupInsight AI Backend Running"

@app.route("/status")
def status():
    return jsonify({"status": "online"})

@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.get_json()
        prompt = data.get("prompt", "").strip()
        if not prompt:
            return jsonify({"error": "No prompt provided"}), 400

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        output_ids = model.generate(input_ids, max_new_tokens=100)
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return jsonify({"output": output_text})
    except Exception as e:
        return jsonify({"error": f"Text generation failed: {str(e)}"}), 500

@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json()
        question = data.get("question", "").strip()
        url = data.get("url", "").strip()

        if not question or not url:
            return jsonify({"error": "Both 'question' and 'url' fields are required."}), 400

        context = scrape_text_from_url(url)
        if not context:
            return jsonify({"error": "Failed to scrape content from URL."}), 500

        answer = qa_pipeline(question=question, context=context)

        return jsonify({
            "question": question,
            "answer": answer.get("answer", "No answer found."),
            "score": answer.get("score", 0.0)
        })
    except Exception as e:
        return jsonify({"error": f"Question answering failed: {str(e)}"}), 500

# Run the app on Render's dynamic port
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)

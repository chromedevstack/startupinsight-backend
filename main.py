from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup
from transformers import pipeline, T5ForConditionalGeneration, T5Tokenizer
import os

app = Flask(__name__)
CORS(app)

# Load Flan-T5 model for text generation
model_name = "google/flan-t5-small"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

# Load lightweight QA pipeline for question answering
qa_pipeline = pipeline("question-answering")

def scrape_text_from_url(url):
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text = ' '.join(p.get_text() for p in paragraphs)
        return text
    except Exception:
        return None

@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.get_json()
        prompt = data.get("prompt", "")
        if not prompt:
            return jsonify({"error": "No prompt provided"}), 400

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        outputs = model.generate(input_ids, max_new_tokens=100)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return jsonify({"output": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = data.get("question")
    url = data.get("url")

    if not question or not url:
        return jsonify({"error": "Both 'question' and 'url' fields are required."}), 400

    scraped_text = scrape_text_from_url(url)
    if not scraped_text:
        return jsonify({"error": "Failed to scrape content from URL."}), 500

    try:
        answer = qa_pipeline(question=question, context=scraped_text)
        return jsonify({
            "question": question,
            "answer": answer.get("answer", "No answer found."),
            "score": answer.get("score", 0)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/status")
def status():
    return jsonify({"status": "online"})

@app.route("/")
def home():
    return "Combined AI backend is running."

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  # Use Render's PORT env or fallback 8080
    app.run(host="0.0.0.0", port=port)

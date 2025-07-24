from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load models only once
print("Loading AI models...")

# --- For Question-Answering ---
# Using a much smaller BERT-Tiny model fine-tuned for SQuAD
# Model size: ~24.34 MB (compared to much larger defaults)
qa_model_name = "mrm8488/bert-tiny-5-finetuned-squadv2"
qa_pipeline = None # Initialize to None in case of loading failure
try:
    # Explicitly set device to 'cpu' for free tier deployments which are typically CPU-only
    qa_pipeline = pipeline("question-answering", model=qa_model_name, device='cpu')
    print(f"Question Answering model '{qa_model_name}' loaded successfully.")
except Exception as e:
    print(f"Error loading QA model '{qa_model_name}': {e}")
    print("Question Answering functionality will be unavailable.")

# --- For Text Generation ---
# Using an extremely tiny LLM for generation (~10 Million parameters)
gen_model_name = "arnir0/Tiny-LLM"
gen_tokenizer = None
gen_model = None
try:
    # Explicitly set device to 'cpu'
    gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
    gen_model = AutoModelForCausalLM.from_pretrained(gen_model_name).to('cpu')
    # Ensure pad_token and eos_token are set if not already present, crucial for generation
    if gen_tokenizer.pad_token is None:
        gen_tokenizer.pad_token = gen_tokenizer.eos_token # Or a specific pad token if available
    print(f"Text Generation model '{gen_model_name}' loaded successfully.")
except Exception as e:
    print(f"Error loading Generation model '{gen_model_name}': {e}")
    print("Text Generation functionality will be unavailable.")

print("All AI models loading attempts complete.")

# Scrape text from a URL
def scrape_text_from_url(url):
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        # Prioritize content within common article/main tags
        main_content = soup.find(['article', 'main', 'body']) 
        if main_content:
            text = ' '.join(p.get_text() for p in main_content.find_all('p'))
            if not text: # If no <p> tags in main content, try other block elements
                text = ' '.join(main_content.get_text(separator=' ', strip=True).split())
        else: # Fallback to body if no specific content tags found
            text = ' '.join(p.get_text() for p in soup.find_all('p'))
            if not text:
                text = ' '.join(soup.get_text(separator=' ', strip=True).split())

        return text.strip()
    except requests.exceptions.RequestException as req_e:
        print(f"HTTP/Network error during scraping: {req_e}")
        return None
    except Exception as e:
        print(f"General scraping error: {e}")
        return None

# --- Routes ---
@app.route("/")
def home():
    return "StartupInsight AI Backend Running"

@app.route("/status")
def status():
    # Check if models were loaded successfully
    qa_status = "loaded" if qa_pipeline else "failed"
    gen_status = "loaded" if gen_model and gen_tokenizer else "failed"
    overall_status = "online" if qa_pipeline or (gen_model and gen_tokenizer) else "degraded (no models)"
    return jsonify({
        "status": overall_status,
        "qa_model_status": qa_status,
        "gen_model_status": gen_status
    })

@app.route("/generate", methods=["POST"])
def generate():
    try:
        if not gen_model or not gen_tokenizer:
            return jsonify({"error": "Text generation model not loaded. Please check backend status."}), 500

        data = request.get_json()
        prompt = data.get("prompt", "").strip()
        if not prompt:
            return jsonify({"error": "No prompt provided"}), 400

        input_ids = gen_tokenizer(prompt, return_tensors="pt").input_ids
        # Ensure input tensor is on CPU
        input_ids = input_ids.to('cpu')

        # Adjust generation parameters for better results with tiny models
        output_ids = gen_model.generate(
            input_ids,
            max_new_tokens=50, # Keep output short to save computation and memory
            do_sample=True,    # Use sampling for more varied output
            top_k=50,          # Limit to top K words
            top_p=0.95,        # Use nucleus sampling
            temperature=0.7,   # Control randomness
            # Using pad_token_id for eos_token_id often helps stop generation cleanly
            eos_token_id=gen_tokenizer.eos_token_id if gen_tokenizer.eos_token_id is not None else gen_tokenizer.pad_token_id,
            pad_token_id=gen_tokenizer.pad_token_id if gen_tokenizer.pad_token_id is not None else gen_tokenizer.eos_token_id
        )
        output_text = gen_tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Remove the input prompt from the output if it's an appended generation
        if output_text.startswith(prompt):
            output_text = output_text[len(prompt):].strip()

        return jsonify({"output": output_text})
    except Exception as e:
        return jsonify({"error": f"Text generation failed: {str(e)}"}), 500

@app.route("/ask", methods=["POST"])
def ask():
    try:
        if not qa_pipeline:
            return jsonify({"error": "Question answering model not loaded. Please check backend status."}), 500

        data = request.get_json()
        question = data.get("question", "").strip()
        url = data.get("url", "").strip()

        if not question or not url:
            return jsonify({"error": "Both 'question' and 'url' fields are required."}), 400

        context = scrape_text_from_url(url)
        if not context:
            return jsonify({"error": "Failed to scrape content from URL or no relevant text found."}), 500
        
        # Limit context size for smaller models
        # QA models typically have max sequence lengths (e.g., 512 tokens).
        # Truncating context helps prevent errors and OOM for large inputs.
        # This is a basic truncation, for production, consider more intelligent chunking.
        max_context_length_tokens = 400 # A common safe max length for small BERT-like models
        # A rough token estimate by splitting words; for true tokenization, use model's tokenizer.
        # But this is a quick way to reduce input size.
        words = context.split()
        if len(words) > max_context_length_tokens:
            context = ' '.join(words[:max_context_length_tokens])
            print(f"Context truncated to approx. {max_context_length_tokens} words.")

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
    # In a production environment like Render, set debug=False
    app.run(host="0.0.0.0", port=port, debug=False)

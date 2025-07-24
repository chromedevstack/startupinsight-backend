import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoModelForQuestionAnswering
import torch

# Initialize Flask app
app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# Global variables for models and their status
qa_pipeline = None
generator_pipeline = None
qa_model_status = "unloaded"
gen_model_status = "unloaded"

# Determine device (CPU for free tiers)
# Using "cpu" explicitly as Render's free tier typically does not offer GPUs
device = "cpu"
print(f"Device set to use {device}")

# Function to load AI models
def load_ai_models():
    global qa_pipeline, generator_pipeline, qa_model_status, gen_model_status

    print("Loading AI models...")

    # --- Load Question Answering Model ---
    qa_model_name = "mrm8488/bert-tiny-5-finetuned-squadv2"
    try:
        # Load tokenizer and model
        qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
        qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)

        # Create question-answering pipeline
        qa_pipeline = pipeline("question-answering", model=qa_model, tokenizer=qa_tokenizer, device=device)
        print(f"Question Answering model '{qa_model_name}' loaded successfully.")
        qa_model_status = "loaded"
    except Exception as e:
        print(f"Error loading Question Answering model '{qa_model_name}': {e}")
        qa_model_status = "failed"
        # Optional: Log full traceback for debugging
        import traceback
        traceback.print_exc()

    # --- Load Generation Model ---
    gen_model_name = "arnir0/Tiny-LLM"
    try:
        # Load tokenizer and model
        gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
        # Note: If memory issues persist, this is the line you'd modify for quantization (e.g., load_in_8bit=True)
        gen_model = AutoModelForCausalLM.from_pretrained(gen_model_name)

        # Create text-generation pipeline
        generator_pipeline = pipeline("text-generation", model=gen_model, tokenizer=gen_tokenizer, device=device)
        print(f"Generation model '{gen_model_name}' loaded successfully.")
        gen_model_status = "loaded"
    except Exception as e:
        print(f"Error loading Generation model '{gen_model_name}': {e}")
        gen_model_status = "failed"
        print("Text Generation functionality will be unavailable.")
        # Optional: Log full traceback for debugging
        import traceback
        traceback.print_exc()

    print("All AI models loading attempts complete.")

# Load models when the application starts
# This will execute as soon as `python main.py` is run
load_ai_models()

# --- API Endpoints ---

@app.route('/')
def home():
    """Basic endpoint to confirm the server is running."""
    return jsonify({"message": "Backend is online and ready.", "status": "online",
                    "qa_model_status": qa_model_status,
                    "gen_model_status": gen_model_status})

@app.route('/status')
def status():
    """Endpoint to check the status of the backend and models."""
    return jsonify({
        "status": "online" if qa_model_status == "loaded" and gen_model_status == "loaded" else "degraded",
        "qa_model_status": qa_model_status,
        "gen_model_status": gen_model_status
    })

@app.route('/ask', methods=['POST'])
def ask_question():
    """Endpoint for the Question Answering model."""
    if qa_pipeline is None or qa_model_status == "failed":
        return jsonify({"error": "Question Answering model not loaded.", "qa_model_status": qa_model_status}), 503

    data = request.json
    question = data.get('question')
    context_url = data.get('url') # URL from frontend

    if not question or not context_url:
        return jsonify({"error": "Question and URL are required."}), 400

    try:
        # Fetch context from URL (simplified, ideally more robust parsing)
        # Note: This context fetching is basic. For production, consider a robust web scraping library or API.
        import requests
        from bs4 import BeautifulSoup
        response = requests.get(context_url, timeout=10)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract text content, prioritizing common content tags
        paragraphs = soup.find_all('p')
        content_divs = soup.find_all('div', class_=lambda c: c and ('content' in c or 'main' in c))
        
        context_text = ""
        if paragraphs:
            context_text = "\n".join([p.get_text() for p in paragraphs])
        elif content_divs:
            context_text = "\n".join([d.get_text() for d in content_divs])
        else:
            context_text = soup.get_text() # Fallback to all text

        if len(context_text) > 5000: # Limit context size for smaller models
            context_text = context_text[:5000]

        if not context_text.strip():
            return jsonify({"error": "Could not extract sufficient text from the provided URL. The page might be empty, JavaScript-heavy, or blocked."}), 400


        # Use QA pipeline
        result = qa_pipeline(question=question, context=context_text)
        return jsonify(result) # result contains 'answer', 'score', 'start', 'end'
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Failed to access URL: {e}"}), 400
    except Exception as e:
        return jsonify({"error": f"An error occurred during QA: {e}"}), 500

@app.route('/generate', methods=['POST'])
def generate_text():
    """Endpoint for the Text Generation model."""
    if generator_pipeline is None or gen_model_status == "failed":
        return jsonify({"error": "Generation model not loaded.", "gen_model_status": gen_model_status}), 503

    data = request.json
    prompt = data.get('prompt')

    if not prompt:
        return jsonify({"error": "Prompt is required."}), 400

    try:
        # Use generation pipeline
        # Max new tokens for a short, concise response
        # Adjust based on desired output length and model capabilities
        result = generator_pipeline(prompt, max_new_tokens=200, num_return_sequences=1)
        # The result is a list of dictionaries, take the generated_text from the first one
        generated_text = result[0]['generated_text']

        # Post-process to remove the original prompt if the model echoes it
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()

        return jsonify({"output": generated_text})
    except Exception as e:
        return jsonify({"error": f"An error occurred during text generation: {e}"}), 500

# --- Main execution block ---
if __name__ == '__main__':
    # Get the port from the environment variable provided by Render (e.g., 10000)
    # Default to 5000 for local testing if PORT is not set
    port = int(os.environ.get('PORT', 5000))
    # Run the Flask app
    app.run(host='0.0.0.0', port=port, debug=False)

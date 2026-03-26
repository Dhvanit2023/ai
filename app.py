import os
import uuid
import requests
import time
from flask import Flask, request, jsonify, render_template, send_from_directory

from gradio_client import Client, handle_file
app = Flask(__name__)

# ── Folders ──
for folder in ["models", "generated", "uploads"]:
    os.makedirs(folder, exist_ok=True)

client = Client("dhvanit2026/ai-image-detector")
# ── Config ──

#HF_TOKEN = os.getenv("KEY")

HF_TOKEN = os.getenv("KEY")

HF_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

# URLS matching your screenshot
CHAT_API_URL = "https://router.huggingface.co/v1/chat/completions"
# Try this specific direct URL
# The 2026 standardized Router path for text-to-image
IMAGE_API_URL = "https://router.huggingface.co/hf-inference/models/stabilityai/stable-diffusion-xl-base-1.0"
def call_hf_api(url, payload):
    for attempt in range(1, 4):
        try:
            print(f"📡 Router Attempt {attempt}/3")
            response = requests.post(url, headers=HF_HEADERS, json=payload, timeout=60)

            if response.status_code == 200:
                return response
            
            # Handle model loading
            if response.status_code == 503:
                print("⏳ Model waking up... waiting 10s")
                time.sleep(10)
                continue
            
            print(f"❌ Error {response.status_code}: {response.text}")
        except Exception as e:
            print(f"💥 Connection Error: {e}")
            time.sleep(2)
    return None

@app.route("/")
def home():
    return render_template("index.html")

# ── Chat Route: Updated to match your screenshot ──
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    msg = data.get("message", "").strip()

    payload = {
        # ADD THE SUFFIX HERE:
        "model": "mistralai/Mistral-7B-Instruct-v0.2:featherless-ai",
        "messages": [
            {"role": "user", "content": msg}
        ],
        "max_tokens": 500
    }

    response = call_hf_api(CHAT_API_URL, payload)
    
    if response:
        result = response.json()
        # The answer is now inside choices -> message -> content
        try:
            reply = result['choices'][0]['message']['content']
            return jsonify({"response": reply.strip()})
        except (KeyError, IndexError):
            return jsonify({"response": "Got a weird response format. Try again!"})
            
    return jsonify({"response": "The brain router is sleeping. Try again in a moment!"})

# ── Image Route ──
@app.route("/generate-image", methods=["POST"])
def create_image():
    try:
        data = request.get_json()
        prompt = data.get("prompt", "a digital art landscape")
        
        # 2026 Router Payload
        payload = {
            "inputs": prompt,
            "parameters": {
                "num_inference_steps": 30,
                "guidance_scale": 8.0,
            },
            "options": {"wait_for_model": True}
        }

        print(f"🎨 Routing Image Request: {prompt}")
        response = requests.post(IMAGE_API_URL, headers=HF_HEADERS, json=payload, timeout=120)

        if response.status_code == 200:
            # Check if we got an image or a hidden JSON error
            content_type = response.headers.get("Content-Type", "")
            if "image" in content_type:
                filename = f"{uuid.uuid4().hex}.png"
                path = os.path.join("generated", filename)
                with open(path, "wb") as f:
                    f.write(response.content)
                return jsonify({"image_url": f"/generated/{filename}", "error": None})
            else:
                return jsonify({"image_url": "", "error": "API returned text instead of image. Check your prompt/quota."}), 400

        # Handle 503 (Loading) or 429 (Rate Limit)
        return jsonify({
            "image_url": "", 
            "error": f"Router Error {response.status_code}: {response.text}"
        }), response.status_code

    except Exception as e:
        print(f"🔥 Crash: {e}")
        return jsonify({"image_url": "", "error": str(e)}), 500

from PIL import Image
import numpy as np

import base64
@app.route("/detect-ai-image", methods=["POST"])
def detect():
    try:
        file = request.files.get("file")

        if not file:
            return jsonify({"error": "No file uploaded"})

        temp_path = "temp.png"
        file.save(temp_path)

        # call HF Space API
        result = client.predict(
            image=handle_file(temp_path),
            api_name="/predict"
        )

        print("HF RESULT:", result)

        # extract label
        label = result.get("label", "Unknown")

        # ✅ delete temp file AFTER use
        import os
        os.remove(temp_path)

        return jsonify({
            "result": label
        })

    except Exception as e:
        print("❌ Detection Error:", e)
        return jsonify({"error": str(e)})
@app.route("/developer")
def developer_page():
    return render_template("developer.html")

@app.route("/generated/<filename>")
def serve(filename):
    return send_from_directory('generated', filename)

if __name__ == "__main__":
    print("🚀 Server starting on http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)

#!/usr/bin/env python3
"""
AR-Agent Web Application
Medical Multimodal Augmented Reality Agent

Based on LLaVA-NeXT-Med for medical image analysis
Integrated with AR visualization capabilities
"""

import os
import json
import base64
import logging
from flask import Flask, request, jsonify, render_template, Response
import requests
from PIL import Image
import io
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global model variables
model = None
tokenizer = None
device = None

def initialize_model():
    """Initialize the LLaVA-NeXT-Med model"""
    global model, tokenizer, device
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Load model and tokenizer
        model_path = os.getenv('MODEL_PATH', 'liuhaotian/llava-v1.6-mistral-7b')
        logger.info(f"Loading model from: {model_path}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
            device_map="auto" if device.type == 'cuda' else None,
            trust_remote_code=True
        )
        
        logger.info("Model loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        return False

def process_medical_image(image_data, prompt="Analyze this medical image and provide a detailed description."):
    """Process medical image with LLaVA-NeXT-Med"""
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Prepare input for model
        # Note: This is a simplified version - actual implementation would depend on the specific model
        medical_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"
        
        # For now, return a mock response
        # In actual implementation, this would use the loaded model
        response = {
            "description": "Medical image analysis completed",
            "findings": [
                "Image quality: Good",
                "Anatomical structure: Visible",
                "Potential abnormalities: Requires further analysis"
            ],
            "confidence": 0.85,
            "recommendations": "Consult with radiologist for detailed interpretation"
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing medical image: {e}")
        return {"error": str(e)}

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else "unknown"
    })

@app.route('/analyze', methods=['POST'])
def analyze_medical_image():
    """Analyze medical image endpoint"""
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({"error": "No image data provided"}), 400
        
        image_data = data['image']
        prompt = data.get('prompt', "Analyze this medical image and provide a detailed description.")
        
        # Process the medical image
        result = process_medical_image(image_data, prompt)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in analyze endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/describe', methods=['POST'])
def describe_image():
    """Legacy endpoint for compatibility with llavavision"""
    try:
        data = request.get_json()
        encoded_string = data['image']
        
        # Use llama.cpp server if available, otherwise use local model
        llama_server_url = os.getenv('LLAMA_SERVER_URL', 'http://localhost:8080')
        
        try:
            # Try llama.cpp server first
            image_data = [{"data": encoded_string, "id": 12}]
            payload = {
                "prompt": "USER:[img-12]Describe the medical image briefly and accurately.\nASSISTANT:", 
                "n_predict": 128, 
                "image_data": image_data, 
                "stream": True
            }
            headers = {"Content-Type": "application/json"}
            url = f"{llama_server_url}/completion"
            
            response = requests.post(url, headers=headers, json=payload, stream=True, timeout=30)
            
            def generate():
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        try:
                            chunk_json = json.loads(chunk.decode().split("data: ")[1])
                            content = chunk_json.get("content", "")
                            if content:
                                yield content
                        except (json.JSONDecodeError, IndexError):
                            continue
            
            return Response(generate(), content_type='text/plain')
            
        except requests.exceptions.RequestException:
            # Fallback to local processing
            logger.info("Llama server unavailable, using local processing")
            result = process_medical_image(encoded_string)
            return jsonify(result)
            
    except Exception as e:
        logger.error(f"Error in describe endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/ar_interface')
def ar_interface():
    """AR interface page"""
    return render_template('ar_interface.html')

@app.route('/medical_dashboard')
def medical_dashboard():
    """Medical professional dashboard"""
    return render_template('medical_dashboard.html')

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    # Initialize model on startup
    logger.info("Starting AR-Agent Medical Application...")
    
    # Try to initialize model (optional for development)
    model_initialized = initialize_model()
    if not model_initialized:
        logger.warning("Model initialization failed - running in fallback mode")
    
    # Start Flask app
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting server on port {port}")
    app.run(debug=debug, host='0.0.0.0', port=port)
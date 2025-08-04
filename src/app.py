#!/usr/bin/gp2_env python
# app.py
"""
This script launches the Flask application for the GPT-2 pre-trained text generator.
Provides a RESTful API endpoint for generating humorous responses to text prompts.
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import necessary dependencies
from flask import Flask, request, jsonify
from src.send_prompt import get_prompt

# Initialize Flask application
app = Flask(__name__)

# Flask route for sending prompts and receiving generated responses
@app.route('/send_prompt', methods=['POST'])
def api_send_prompt():
    """
    API endpoint that accepts POST requests with a text prompt and returns a generated response.
    
    Expected JSON payload:
    {
        "prompt": "Your question or joke here"
    }
    
    Returns:
    {
        "response": "Generated humorous response"
    }
    """
    data = request.json
    prompt = data.get('prompt', '')
    response = get_prompt(prompt)
    print(f"====> Prompt: {prompt}")
    print(f"====> Response: {response}")
    return jsonify({"response": response})

# Launch the Flask application
if __name__ == '__main__':
    app.run(debug=True)  # Run Flask application in debug mode
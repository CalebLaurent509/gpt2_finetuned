#! /usr/bin/gpt2_env python3
# send_prompt.py
"""
This script sends a prompt to the GPT-2 pre-trained text generator and handles response processing.
Provides intelligent response generation with fallback mechanisms and quality validation.
"""
# Import necessary dependencies
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import re
import warnings
import random
import json
from src.train import tokenizer
from src.generator import GenerateText

random.seed()
warnings.filterwarnings("ignore", category=FutureWarning)
#----------------------------------------

# Load data from JSON file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
response_path = os.path.abspath(os.path.join(BASE_DIR, "data", "responses_custom.json"))
with open(response_path, "r", encoding="utf-8") as json_file:
    data = json.load(json_file)
    default_responses = data["default_responses"]
    adaptive_phrases = data["adaptive_phrases"]

# Function to send a prompt to the generator
def get_prompt(prompt=None):
    """
    Sends a prompt to the GPT-2 pre-trained text generator and returns the generated response.

    Args:
        prompt (str): The input prompt to be sent to the text generator.
    
    Returns:
        str: The generated response from the text generator, with the first sentence or question extracted.
             If generation fails or produces low-quality output, returns a random default response.
    """

    # If prompt is not provided, ask user to enter it
    if prompt is None:
        prompt = input("Enter the Prompt: ")
    
    # Remove extra spaces before question marks if any
    prompt = re.sub(r'\s+\?', '?', prompt)
    
    # Add question mark if prompt doesn't contain one
    if '?' not in prompt:
        prompt += '?'
    
    # Initialize the generator
    generator = GenerateText(prompt) 
    generator.input_fn()  # Prepare input for generator
    encoded_output = generator.output_fn()  # Generate output
    
    # Decode the encoded output to readable text
    output = tokenizer.decode(encoded_output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    
    # Extract the response portion after the question mark
    joke_response_start, joke_response_end = output.find('?'), output.find('.')
    
    # Get the first sentence or question from the response
    joke_response = output[joke_response_start + 1:joke_response_end + 1]
    
    # Quality validation conditions
    condition1 = len(joke_response) > 1  # Contains at least 2 characters
    condition2 = len(joke_response) < 100  # Contains less than 100 characters
    
    # Validate response quality
    if condition1 and condition2:
        # Add adaptive phrase to enhance personality
        joke_response += " " + random.choice(adaptive_phrases)
        return joke_response
    else:
        # Return default response if generation quality is poor
        return random.choice(default_responses)

if __name__ == '__main__':
    # Launch the application if no arguments are passed
    model_response = get_prompt(input("Enter the Prompt: "))
    print(model_response)
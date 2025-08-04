#!/usr/bin/gpt2_env python
# generator.py
"""
This script handles text generation using the fine-tuned model.
Provides the core text generation functionality with customizable parameters.
"""
# Import necessary dependencies
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import GPT2LMHeadModel
from src.train import trainer, tokenizer
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

#----------------------------------------

# Model saving functionality
# Path to save the model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pretrained_path = os.path.abspath(os.path.join(BASE_DIR, "src", "models", "model_saved"))

# Check if model doesn't exist in the 'model_saved' directory
if not os.listdir(pretrained_path):
    trainer.save_model(pretrained_path)
    print("Model has been saved.")
else:
    print("Model already exists, no action needed.")

# Load the pre-trained GPT2 model
pretrained_model = GPT2LMHeadModel.from_pretrained(pretrained_path)


# Class for generating text using the model
class GenerateText:
    """
    Generates text using a pre-trained GPT-2 language model.
    
    The GenerateText class provides a way to generate text using the pre-trained GPT-2 language model.
    It takes a prompt as input and generates text based on that prompt.
    
    Args:
        prompt (str): The input prompt to use for text generation.
    
    Attributes:
        prompt (str): The input prompt.
        input_ids (torch.Tensor): The token IDs for the language model input.
        attention_mask (torch.Tensor): The attention mask for input tokens.
        DEFAULT_ENCODE_PARAMS (dict): Default parameters for encoding the input prompt.

    Methods:
        input_fn(): Encodes the input prompt and stores token IDs and attention mask.
        output_fn(): Generates text using the pre-trained GPT-2 language model.
    """
        
    def __init__(self, prompt):
        """
        Initialize the text generator with a prompt.
        
        Args:
            prompt (str): The text prompt to generate from
        """
        self.prompt = prompt
        self.input_ids = None
        self.attention_mask = None
        self.DEFAULT_ENCODE_PARAMS = {
            'return_tensors': 'pt',
            'padding': True,
            'truncation': True,
            'max_length': 300,
        }
    
    def input_fn(self):
        """
        Encode the input prompt and prepare it for model input.
        Sets up tokenized input with proper padding and attention masks.
        """
        tokenizer.pad_token = tokenizer.eos_token  # Set padding token
        input = tokenizer.encode_plus(
            text=self.prompt,
            **self.DEFAULT_ENCODE_PARAMS  # Use default parameters
        )
        self.input_ids, self.attention_mask  = input['input_ids'], input['attention_mask']
    
    def output_fn(self):
        """
        Generate text output using the fine-tuned GPT-2 model.
        
        Returns:
            torch.Tensor: Generated token sequences
        """
        output = pretrained_model.generate(
            self.input_ids,
            attention_mask=self.attention_mask,  # Use attention mask
            max_length=60,  # Set maximum length of generated text
            num_beams=5,  # Number of beams for beam search generation
            no_repeat_ngram_size=5,  # Size of n-gram window to exclude repetitions
            pad_token_id=tokenizer.eos_token_id,  # Padding token ID
            temperature=0.5,  # Temperature for text generation control
            do_sample=True,  # Enable sampling for text generation
            top_p=0.9,  # Cumulative probability threshold for top-p sampling
        )
        return output
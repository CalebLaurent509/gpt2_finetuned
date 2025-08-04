#!/usr/bin/gpt2_env python
# make_dataset.py
"""
This script loads training and validation data and creates a dataset for GPT-2 model training.
Handles dataset creation, tokenization, and data collation for efficient training.
"""
# Import necessary dependencies
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import GPT2Tokenizer
from transformers import TextDataset, DataCollatorForLanguageModeling
#----------------------------------------

# Path to preprocessed data files
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
train_path = os.path.join(BASE_DIR, "data", "preprocessed_data", "train_dataset.txt")
test_path = os.path.join(BASE_DIR, "data", "preprocessed_data", "test_dataset.txt")

# Load the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Maximum block size (number of characters per block)
block_size = 128

# Function to load the dataset
def load_dataset(train_path, test_path, tokenizer):
    """
    This function loads the dataset and prepares it for fine-tuning.
    It takes the path to the train file and the tokenizer as input.
    Returns the training dataset, test dataset, and data collator.
    
    Args:
        train_path (str): Path to training data file
        test_path (str): Path to test data file  
        tokenizer: GPT-2 tokenizer instance
        
    Returns:
        tuple: (train_dataset, test_dataset, data_collator)
    """

    # Create training and validation datasets
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=train_path,
        block_size=block_size
    )
    test_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=test_path,
        block_size=block_size
    )
    
    # Group and format data into batches of 128 characters
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Set to False for causal language modeling (GPT-2 style)
    )
    return train_dataset, test_dataset, data_collator

# Load datasets and data collator
train_dataset, test_dataset, data_collator = load_dataset(train_path, test_path, tokenizer)
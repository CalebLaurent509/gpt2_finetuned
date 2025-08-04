#!/usr/bin/gpt2_env python3
# preprocessing.py
"""
This script performs data preprocessing for GPT-2 model training.
Handles data cleaning, deduplication, and preparation for training datasets.
"""
# Import necessary dependencies
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import re
import pandas as pd
from sklearn.model_selection import train_test_split
#----------------------------------------

# Path to CSV file containing main raw data
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
raw_data_path = os.path.abspath(os.path.join(BASE_DIR, "data", "raw_data", "shortjokes.csv"))
df = pd.read_csv(raw_data_path)

# Path to CSV file containing external data
external_df = pd.read_csv(os.path.abspath(os.path.join(BASE_DIR, "data", "external_data", "jokes.csv")))
external_df.drop(columns=['Question', 'Answer'], inplace=True)  # Remove unnecessary columns

# Concatenate main raw data with external data
df_concat = pd.concat([df, external_df], ignore_index=True)
df_concat = df.drop_duplicates()  # Remove duplicates
df_concat = df.dropna()   # Remove rows with missing values

# Function to clean the data
def cleaning(s):
    """
    This function preprocesses data by removing special characters, numbers, and unnecessary spaces.
    It takes a string as input and returns a cleaned string.
    
    Args:
        s (str): Input string to be cleaned
        
    Returns:
        str: Cleaned string with normalized formatting
    """
        
    s = str(s)
    s = re.sub(r'\s\W',' ', s)  # Remove special characters preceded by whitespace
    s = re.sub(r'\W,\s',' ', s)  # Remove special characters followed by comma and space
    s = re.sub(r"\d+", "", s)  # Remove all digits
    s = re.sub(r'\s+',' ', s)  # Replace multiple spaces with single space
    s = re.sub(r'[!@#$_]', '', s)  # Remove specific special characters
    s = s.replace(r"co","")  # Remove 'co' substring
    s = s.replace(r"https","")  # Remove https
    s = s.replace(r"http","")  # Remove http
    s = s.replace(r"www.","")  # Remove www.
    s = s.replace(r"://","")  # Remove ://
    s = s.replace(r"[\w*"," ")  # Replace certain patterns with space
    return s

# Function to build text files
def build_text_files(data: pd.DataFrame, filename: str) -> None:
    """
    Writes cleaned content from a DataFrame to a text file.

    Args:
        data (pd.DataFrame): DataFrame containing jokes to write
        filename (str): Output file name

    Raises:
        IOError: If an error occurs during file writing
    """
    try:
        with open(filename, 'w', encoding='utf-8') as text_data:
            for _, item in data.iterrows():
                cleaned_joke = cleaning(item['Joke'])
                text_data.write(cleaned_joke)
    except IOError as e:
        print(f"Error writing file: {e}")
        raise

# Split data into training and validation sets
train_data, test_data = train_test_split(df_concat, test_size=0.2, random_state=42)

# Build text files for training and validation sets
preprocessed_train_data_path = os.path.abspath(os.path.join(BASE_DIR, "data", "preprocessed_data", "train_dataset.txt"))
preprocessed_test_data_path = os.path.abspath(os.path.join(BASE_DIR, "data", "preprocessed_data", "test_dataset.txt"))

build_text_files(train_data, preprocessed_train_data_path)
build_text_files(test_data, preprocessed_test_data_path)

print("Train data length:", len(train_data))
print("Test data length:", len(test_data))

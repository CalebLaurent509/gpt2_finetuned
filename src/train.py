#!/usr/bin/gpt2_env python
# train.py
"""
This script performs the training of the GPT-2 model using custom joke datasets.
Implements fine-tuning with comprehensive evaluation and metrics logging.
"""
# Import necessary dependencies
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.make_dataset import train_dataset, test_dataset, data_collator
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

#----------------------------------------
# GPT-2 Model Implementation
model_name = "gpt2"  # Name of the pre-trained GPT-2 model to load
# Load the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
# Load the pre-trained GPT-2 model
model = GPT2LMHeadModel.from_pretrained(model_name)

# Initialize training variables
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
train_dir = os.path.abspath(os.path.join(BASE_DIR, "src", "models", "training_steps"))
train_epochs = 10
train_batch_size = 32
eval_batch_size = 64
eval_steps = 400
save_steps = 1600
warmup_steps = 500
save_limit = 3

# Training arguments for model training configuration
training_args = TrainingArguments(
    output_dir=train_dir,   # Directory to save training states
    overwrite_output_dir=True,   # Overwrite output directory contents if already present
    num_train_epochs=train_epochs,   # Number of training epochs
    per_device_train_batch_size=train_batch_size,   # Training batch size
    per_device_eval_batch_size=eval_batch_size,   # Evaluation batch size
    save_steps=save_steps,   # Number of steps after which the model is saved
    eval_steps=eval_steps,   # Number of steps between evaluations
    warmup_steps=warmup_steps,   # Number of warmup steps for learning rate scheduler
    prediction_loss_only=True,  # Calculate only prediction loss
    save_total_limit=save_limit   # Maximum number of saved models
)

# Model training process
trainer = Trainer(
    model=model,   # Model to train
    args=training_args,   # Training arguments
    data_collator=data_collator,  # Data collator for data loading
    train_dataset=train_dataset,   # Training dataset
    eval_dataset=test_dataset   # Evaluation dataset
)

# Model training execution
if __name__ == "__main__":
    """
    Note:
        Model training can take time depending on available computational power (approx. 30 minutes on Colab).
        Once training is complete, the model will be saved in the directory specified in training_args.
        The trained model can then be used to generate jokes from starting words.
    """
    trainer.train()

    # Evaluate the model on training and test datasets
    # Save evaluation metrics
    # Create a .logs folder for metrics
    log_dir = "./src/visualisation/.logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Save metrics to log file
    def save_metrics(metrics, split_name):
        """
        Save evaluation metrics to log files for performance tracking.
        
        Args:
            metrics (dict): Dictionary containing evaluation metrics
            split_name (str): Name of the dataset split ('train' or 'test')
        """
        log_file = os.path.join(log_dir, f"{split_name}_metrics.log")
        with open(log_file, "a") as f:
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")

    # Evaluate and save metrics
    train_metrics = trainer.evaluate(trainer.train_dataset)
    test_metrics = trainer.evaluate(trainer.eval_dataset)

    # Save to visualisation/.logs/train_metrics.log and visualisation/.logs/test_metrics.log
    save_metrics(train_metrics, "train")
    save_metrics(test_metrics, "test")
    print("Metrics saved to .logs folder.")
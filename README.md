# GPT-2 Fine-Tuned Joke Generator

A sophisticated natural language processing project that fine-tunes OpenAI's GPT-2 model to generate intelligent and humorous responses to jokes, riddles, and questions. This project demonstrates advanced techniques in transformer model fine-tuning, custom dataset preparation, and deploying AI models through a Flask web API.

## Table of Contents
1. [Features](#features)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Data Pipeline](#data-pipeline)
6. [Model Architecture](#model-architecture)
7. [Training Process](#training-process)
8. [API Documentation](#api-documentation)
9. [Evaluation](#evaluation)
10. [Contributing](#contributing)
11. [License](#license)
12. [Contact](#contact)

## Features

- **Custom GPT-2 Fine-Tuning**: Specialized training on joke and riddle datasets
- **Intelligent Response Generation**: Context-aware humor generation with fallback mechanisms
- **RESTful API**: Flask-based web service for easy integration
- **Robust Data Pipeline**: Comprehensive preprocessing and dataset creation workflow
- **Flexible Architecture**: Modular design supporting easy extension and modification
- **Performance Monitoring**: Built-in evaluation metrics and logging system
- **Production Ready**: Complete deployment setup with proper error handling

## Project Structure

```
gpt2_fine-tuned/
├── data/
│   ├── raw_data/           # Original datasets (CSV files)
│   ├── external_data/      # Additional training data
│   ├── preprocessed_data/  # Cleaned and processed text files
│   └── responses_custom.json # Custom response templates
├── src/
│   ├── __init__.py
│   ├── preprocessing.py    # Data cleaning and preparation
│   ├── make_dataset.py    # Dataset creation for training
│   ├── train.py           # Model training pipeline
│   ├── generator.py       # Text generation core logic
│   ├── send_prompt.py     # Prompt processing and response handling
│   ├── app.py             # Flask API application
│   ├── models/
│   │   ├── model_saved/   # Fine-tuned model storage
│   │   └── training_steps/ # Training checkpoints
│   └── visualisation/
│       └── .logs/         # Training metrics and logs
├── notebooks/
│   └── analyse.ipynb      # Data analysis and experimentation
├── requirements.txt       # Python dependencies
├── setup.py              # Package installation configuration
├── .gitignore            # Git ignore rules
└── README.md             # Project documentation
```

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training)
- At least 8GB RAM
- 2GB free disk space for models and data

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/CalebLaurent509/gpt2_fine-tuned.git
   cd gpt2_fine-tuned
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv gpt2_env
   source gpt2_env/bin/activate  # On Windows: gpt2_env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install as a package** (Optional)
   ```bash
   pip install -e .
   ```

## Usage

### Quick Start

1. **Prepare the dataset**
   ```bash
   python src/preprocessing.py
   python src/make_dataset.py
   ```

2. **Train the model**
   ```bash
   python src/train.py
   ```

3. **Generate responses**
   ```bash
   python src/send_prompt.py
   ```

4. **Start the web API**
   ```bash
   python src/app.py
   ```

### Command Line Interface

If installed as a package, you can use the command line interface:
```bash
generate-text "Why did the chicken cross the road?"
```

### API Usage

Send POST requests to the Flask API:
```bash
curl -X POST http://localhost:5000/send_prompt \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What do you call a bear with no teeth?"}'
```

## Data Pipeline

### Data Sources
- **Primary Dataset**: Short jokes from Kaggle
- **External Dataset**: Additional joke collections
- **Custom Responses**: Handcrafted fallback responses and adaptive phrases

### Preprocessing Steps
1. **Data Cleaning**: Remove special characters, URLs, and unnecessary whitespace
2. **Deduplication**: Eliminate duplicate entries across datasets
3. **Quality Filtering**: Remove low-quality or inappropriate content
4. **Text Normalization**: Standardize formatting and encoding
5. **Train/Test Split**: 80/20 split with stratified sampling

### Dataset Statistics
- **Training Set**: ~80% of cleaned data
- **Test Set**: ~20% of cleaned data
- **Block Size**: 128 tokens per training example
- **Vocabulary**: GPT-2 standard tokenizer (50,257 tokens)

## Model Architecture

### Base Model
- **Architecture**: GPT-2 (Generative Pre-trained Transformer 2)
- **Parameters**: 124M parameters
- **Context Length**: 1024 tokens
- **Attention Heads**: 12
- **Hidden Size**: 768

### Fine-Tuning Configuration
- **Learning Rate**: Dynamic with warmup
- **Batch Size**: 32 (training), 64 (evaluation)
- **Epochs**: 10
- **Optimization**: AdamW optimizer
- **Regularization**: Dropout and weight decay

### Generation Parameters
- **Max Length**: 60 tokens
- **Temperature**: 0.5 (balanced creativity/coherence)
- **Top-p Sampling**: 0.9
- **Beam Search**: 5 beams
- **No Repeat N-gram**: 5 (prevent repetition)

## Training Process

### Training Configuration
```python
# Training hyperparameters
EPOCHS = 10
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 64
LEARNING_RATE = 5e-5
WARMUP_STEPS = 500
SAVE_STEPS = 1600
EVAL_STEPS = 400
```

### Training Workflow
1. **Data Loading**: Load preprocessed text datasets
2. **Tokenization**: Convert text to model-compatible tokens
3. **Fine-Tuning**: Adjust pre-trained weights on joke data
4. **Validation**: Regular evaluation on test set
5. **Checkpointing**: Save model state at regular intervals
6. **Metrics Logging**: Track loss, perplexity, and generation quality

### Performance Monitoring
- Training and validation metrics logged to `src/visualisation/.logs/`
- Real-time loss tracking during training
- Automated model checkpointing
- Memory usage optimization

## API Documentation

### Endpoints

#### POST /send_prompt
Generate a humorous response to a given prompt.

**Request Body:**
```json
{
  "prompt": "Your joke or question here"
}
```

**Response:**
```json
{
  "response": "Generated humorous response"
}
```

**Example:**
```bash
curl -X POST http://localhost:5000/send_prompt \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What did the ocean say to the beach?"}'
```

### Response Logic
1. **Input Validation**: Check prompt format and content
2. **Text Generation**: Use fine-tuned model for response generation
3. **Quality Assessment**: Evaluate response appropriateness
4. **Fallback Mechanism**: Use custom responses if generation fails
5. **Response Enhancement**: Add adaptive phrases for personality

## Evaluation

### Metrics
- **Perplexity**: Measures model confidence in predictions
- **BLEU Score**: Evaluates generated text quality
- **Response Appropriateness**: Manual evaluation of humor quality
- **Inference Speed**: Response generation time

### Evaluation Results
Training metrics are automatically saved to:
- `src/visualisation/.logs/train_metrics.log`
- `src/visualisation/.logs/test_metrics.log`

### Benchmarking
The model is evaluated against:
- Baseline GPT-2 responses
- Human-written joke responses
- Other fine-tuned humor models

## Contributing

We welcome contributions to improve the project. Please follow these guidelines:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Commit your changes** (`git commit -m 'Add amazing feature'`)
4. **Push to the branch** (`git push origin feature/amazing-feature`)
5. **Open a Pull Request**

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt
pip install -e .

# Run tests
python -m pytest tests/

# Format code
black src/
isort src/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

**Caleb Laurent**
- Email: laurentcaleb99@gmail.com
- GitHub: [@CalebLaurent509](https://github.com/CalebLaurent509)
- Project Link: [https://github.com/CalebLaurent509/gpt2_fine-tuned](https://github.com/CalebLaurent509/gpt2_fine-tuned)

## Acknowledgments

- OpenAI for the GPT-2 model architecture
- Hugging Face for the Transformers library
- Kaggle community for joke datasets
- Contributors and testers who helped improve the project

---

*Built with passion for AI and humor. Making machines laugh, one joke at a time.*

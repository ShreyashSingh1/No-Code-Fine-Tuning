# No-Code Fine-Tuning

A streamlined web application for fine-tuning Hugging Face models without writing code. This tool allows users to easily fine-tune sentiment classification models and language models (LLMs) through a simple, intuitive interface.

![Landing Page](https://i.imgur.com/OYJTvUb.png)

## Features

- **Two Fine-Tuning Options**:
  - Sentiment Model Fine-Tuning (Classification)
  - LLM Fine-Tuning (Text Generation)

- **Simple 5-Step Process**:
  1. Select fine-tuning type
  2. Provide Hugging Face model ID
  3. Upload your dataset (CSV or JSON)
  4. Define hyperparameters
  5. Start fine-tuning

- **Advanced Capabilities**:
  - Hyperparameter optimization
  - Model evaluation
  - Downloadable fine-tuned models
  - Support for LORA and QLORA techniques

## Installation

```bash
# Clone the repository
git clone  https://github.com/ShreyashSingh1/No-Code-Fine-Tuning.git
cd No-Code-Fine-Tunning

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Run the Streamlit application
python application.py
```

Then open your browser and navigate to http://localhost:8501

### Example Workflow

1. **Select Fine-Tuning Type**: Choose between Sentiment Model Fine-Tuning or LLM Fine-Tuning
2. **Provide Model ID**: Enter a Hugging Face model ID (e.g., `albert-base-v2` for sentiment analysis or `gpt2` for LLM)
3. **Upload Dataset**: Upload your CSV or JSON dataset with appropriate columns
4. **Define Hyperparameters**: Set learning rates, batch sizes, and epochs
5. **Start Fine-Tuning**: Click the "Start Fine-Tuning" button and wait for the process to complete
6. **Download Model**: Once fine-tuning is complete, download your fine-tuned model

## Dataset Format

### For Sentiment Analysis
- CSV or JSON file with text input column and label column
- Example: `{"text": "This movie is great!", "label": 1}`

### For LLM Fine-Tuning
- CSV or JSON file with input text and target output text
- Example: `{"text": "What is machine learning?", "target": "Machine learning is a branch of artificial intelligence..."}`

## Technical Architecture

The application is built using:
- **Streamlit**: For the web interface
- **Hugging Face Transformers**: For model loading and fine-tuning
- **PyTorch**: As the underlying deep learning framework
- **Datasets**: For efficient data handling

The fine-tuning process uses grid search to find optimal hyperparameters and provides detailed training metrics.

## Advanced Features

### LORA and QLORA Support

The application includes support for LORA (Low-Rank Adaptation) and QLORA (Quantized LORA) techniques for efficient fine-tuning of large language models with minimal computational resources.

![Backend Training](https://i.imgur.com/78RbOuD.png)

## References

- [Basics of Fine-Tuning - Shreyash Singh](https://github.com/ShreyashSingh1/Fine-Tuning-models)
- [Advanced Fine-Tuning - Shreyash Singh](https://github.com/ShreyashSingh1/Adavence-Fine-Tunning)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/en/main_classes/processors)
- [QLoRA Fine-Tuning Documentation - Shreyash Singh](https://charmed-amount-e80.notion.site/QLoRA-Fine-Tuning-Documentation-Shreyash-Singh-19f0d537ad5080ec8c62c7ae408911ec)

## License

MIT
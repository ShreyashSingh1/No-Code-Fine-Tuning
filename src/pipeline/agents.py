import os
import zipfile
import tempfile
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from sklearn.model_selection import ParameterGrid
import pandas as pd

class ClassificationAgentTunner:
    def __init__(self, model_id, output_dir):
        self.model_id = model_id
        self.output_dir = output_dir
        self.model, self.tokenizer = self.load_model_and_tokenizer()

    def load_model_and_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_id, num_labels=2)
        return model, tokenizer

    def load_data(self, file, file_format="csv"):
        with tempfile.NamedTemporaryFile(delete=False, mode="wb") as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_file_path = tmp_file.name

        if file_format == "csv":
            dataset = load_dataset("csv", data_files={"train": tmp_file_path})
        elif file_format == "json":
            dataset = load_dataset("json", data_files={"train": tmp_file_path})
        else:
            raise ValueError("Unsupported file format. Use 'csv' or 'json'.")

        return dataset

    def format_data(self, dataset, input_column):
        def preprocess_function(examples):
            return self.tokenizer(examples[input_column], truncation=True, padding="max_length", max_length=128)
        dataset = dataset.map(preprocess_function, batched=True)
        return dataset

    def fine_tune(self, dataset, param_grid):
        best_loss = float("inf")
        best_params = None
        grid = list(ParameterGrid(param_grid))
        for idx, params in enumerate(grid):
            training_args = TrainingArguments(
                output_dir=os.path.join(self.output_dir, f"run_{idx}"),
                evaluation_strategy="epoch",
                save_strategy="epoch",
                learning_rate=params["learning_rate"],
                per_device_train_batch_size=params["batch_size"],
                num_train_epochs=params["epochs"],
                logging_dir="./logs",
                logging_steps=10,
                save_total_limit=1
            )

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=dataset["train"],
                eval_dataset=dataset["train"].select(range(100)),
            )
            trainer.train()
            eval_results = trainer.evaluate()

            if eval_results["eval_loss"] < best_loss:
                best_loss = eval_results["eval_loss"]
                best_params = params
                best_model_dir = os.path.join(self.output_dir, f"run_{idx}")

        return best_params, best_loss, best_model_dir

    def create_zip_from_dir(self, best_model_dir, zip_name="fine_tuned_model.zip"):
        zip_path = os.path.join(self.output_dir, zip_name)
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(best_model_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, best_model_dir)
                    zipf.write(file_path, arcname)
        return zip_path
    
    
    

import os
import tempfile
import zipfile
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
from sklearn.model_selection import ParameterGrid

class LLMAgentTunner:
    def __init__(self, model_id, output_dir):
        self.model_id = model_id
        self.output_dir = output_dir
        self.model, self.tokenizer = self.load_model_and_tokenizer()

    def load_model_and_tokenizer(self):
        # Load the model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        
        # Check if tokenizer has a pad token, if not, set pad_token to eos_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token  # or you can define a custom token like '[PAD]'
        
        model = AutoModelForCausalLM.from_pretrained(self.model_id)
        return model, tokenizer

    def load_data(self, file, file_format="csv"):
        """Load dataset from file (CSV or JSON)."""
        with tempfile.NamedTemporaryFile(delete=False, mode="wb") as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_file_path = tmp_file.name

        # Load dataset based on the file format
        if file_format == "csv":
            dataset = load_dataset("csv", data_files={"train": tmp_file_path})
        elif file_format == "json":
            dataset = load_dataset("json", data_files={"train": tmp_file_path})
        else:
            raise ValueError("Unsupported file format. Use 'csv' or 'json'.")

        return dataset

    def format_data(self, dataset, input_column, target_column=None):
        """
        Preprocess the dataset for text generation.
        If `target_column` is provided, format it for text-to-text tasks.
        """
        def preprocess_function(examples):
            if target_column:
                # Text-to-text tasks: input -> output
                return self.tokenizer(
                    examples[input_column],
                    text_target=examples[target_column],
                    truncation=True,
                    padding="max_length",
                    max_length=128,
                )
            else:
                # Text generation tasks: input only
                return self.tokenizer(
                    examples[input_column],
                    truncation=True,
                    padding="max_length",
                    max_length=128,
                )
        
        # Map preprocessing function to the dataset
        dataset = dataset.map(preprocess_function, batched=True)
        return dataset

    def fine_tune(self, dataset, param_grid):
        """Fine-tune the model using hyperparameter search."""
        best_loss = float("inf")
        best_params = None
        grid = list(ParameterGrid(param_grid))
        for idx, params in enumerate(grid):
            training_args = TrainingArguments(
                output_dir=os.path.join(self.output_dir, f"run_{idx}"),
                evaluation_strategy="epoch",
                save_strategy="epoch",
                learning_rate=params["learning_rate"],
                per_device_train_batch_size=params["batch_size"],
                num_train_epochs=params["epochs"],
                logging_dir="./logs",
                logging_steps=10,
                save_total_limit=1
            )

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=dataset["train"],
                eval_dataset=dataset["train"].select(range(100)),  # For quicker evaluation
            )
            trainer.train()
            eval_results = trainer.evaluate()

            if eval_results["eval_loss"] < best_loss:
                best_loss = eval_results["eval_loss"]
                best_params = params
                best_model_dir = os.path.join(self.output_dir, f"run_{idx}")

        return best_params, best_loss, best_model_dir

    def generate_text(self, prompt, max_length=50):
        """Generate text based on a prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_length=max_length)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def create_zip_from_dir(self, best_model_dir, zip_name="fine_tuned_model.zip"):
        """Zip the best model directory."""
        zip_path = os.path.join(self.output_dir, zip_name)
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(best_model_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, best_model_dir)
                    zipf.write(file_path, arcname)
        return zip_path

# Example usage
if __name__ == "__main__":
    model_id = "gpt2"  # Specify the Hugging Face model ID (e.g., 'gpt2')
    output_dir = "./results"  # Directory to save fine-tuned model

    # Initialize the fine-tuner
    fine_tuner = LLMAgentTunner(model_id, output_dir)

    # Load and preprocess data
    dataset = fine_tuner.load_data(file=open("your_dataset.json", "rb"), file_format="json")
    dataset = fine_tuner.format_data(dataset, input_column="text")

    # Define hyperparameters for fine-tuning
    param_grid = {
        "learning_rate": [5e-5, 1e-4],
        "batch_size": [16, 32],
        "epochs": [3, 5]
    }

    # Fine-tune the model and get the best parameters and model
    best_params, best_loss, best_model_dir = fine_tuner.fine_tune(dataset, param_grid)

    # Generate text using the fine-tuned model
    generated_text = fine_tuner.generate_text("Once upon a time", max_length=100)
    print("Generated Text: ", generated_text)

    # Create a zip of the best model
    zip_path = fine_tuner.create_zip_from_dir(best_model_dir)
    print(f"Fine-tuned model saved as a zip at: {zip_path}")

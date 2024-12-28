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
            print(dataset["train"][0])
        elif file_format == "json":
            dataset = load_dataset("json", data_files={"train": tmp_file_path})
            print(dataset["train"][0])
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
    
    
    
import tempfile
import zipfile
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
from sklearn.model_selection import ParameterGrid
import time

class LLMAgentTunner:
    def __init__(self, model_id, output_dir):
        self.model_id = model_id
        self.output_dir = output_dir
        self.model, self.tokenizer = self.load_model_and_tokenizer()

    def load_model_and_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(self.model_id)
        return model, tokenizer

    def load_data(self, file, file_format="csv"):
        with tempfile.NamedTemporaryFile(delete=False, mode="wb") as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_file_path = tmp_file.name

        if file_format == "csv":
            dataset = load_dataset("csv", data_files={"train": tmp_file_path})
            print(dataset["train"][0])
        elif file_format == "json":
            dataset = load_dataset("json", data_files={"train": tmp_file_path})
            print(dataset["train"][0])
        else:
            raise ValueError("Unsupported file format. Use 'csv' or 'json'.")

        os.remove(tmp_file_path)  
        return dataset

    def format_data(self, dataset, input_column, target_column=None):
        if input_column not in dataset.features:
            raise ValueError(f"{input_column} not found in the dataset.")
        if target_column and target_column not in dataset.features:
            raise ValueError(f"{target_column} not found in the dataset.")

        def preprocess_function(examples):
            if target_column:
                return self.tokenizer(
                    examples[input_column],
                    text_target=examples[target_column],
                    truncation=True,
                    padding="max_length",
                    max_length=128,
                )
            else:
                return self.tokenizer(
                    examples[input_column],
                    truncation=True,
                    padding="max_length",
                    max_length=128,
                )

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

            eval_dataset = dataset.get("validation", dataset["train"].select(range(100)))
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=dataset["train"],
                eval_dataset=eval_dataset,
            )
            trainer.train()
            eval_results = trainer.evaluate()

            if eval_results["eval_loss"] < best_loss:
                best_loss = eval_results["eval_loss"]
                best_params = params
                best_model_dir = os.path.join(self.output_dir, f"run_{idx}")

        return best_params, best_loss, best_model_dir

    def generate_text(self, prompt, max_length=50):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_length=max_length)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def create_zip_from_dir(self, best_model_dir, zip_name="fine_tuned_model.zip"):
        zip_path = os.path.join(self.output_dir, zip_name)
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(best_model_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, best_model_dir)
                    zipf.write(file_path, arcname)
        return zip_path

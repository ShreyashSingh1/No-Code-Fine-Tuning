import streamlit as st
import os
import shutil
import zipfile
import tempfile
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from sklearn.model_selection import ParameterGrid
import pandas as pd
import json

# Function to load model and tokenizer
def load_model_and_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)
    return model, tokenizer

# Function to load data
def load_data(file, file_format="csv"):
    with tempfile.NamedTemporaryFile(delete=False, mode="wb") as tmp_file:
        tmp_file.write(file.getvalue())  # Save the uploaded file content to a temporary file
        tmp_file_path = tmp_file.name
    
    if file_format == "csv":
        dataset = load_dataset("csv", data_files={"train": tmp_file_path})
    elif file_format == "json":
        dataset = load_dataset("json", data_files={"train": tmp_file_path})
    else:
        raise ValueError("Unsupported file format. Use 'csv' or 'json'.")
    
    return dataset

# Function to format data
def format_data(dataset, tokenizer, input_column):
    def preprocess_function(examples):
        return tokenizer(examples[input_column], truncation=True, padding="max_length", max_length=128)
    dataset = dataset.map(preprocess_function, batched=True)
    return dataset

# Function to perform fine-tuning
def fine_tune(model, tokenizer, dataset, param_grid, output_dir):
    best_loss = float("inf")
    best_params = None
    grid = list(ParameterGrid(param_grid))
    for idx, params in enumerate(grid):
        training_args = TrainingArguments(
            output_dir=os.path.join(output_dir, f"run_{idx}"),
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
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["train"].select(range(100)),  # Use a small test set
            tokenizer=tokenizer
        )
        trainer.train()
        eval_results = trainer.evaluate()
        if eval_results["eval_loss"] < best_loss:
            best_loss = eval_results["eval_loss"]
            best_params = params
            best_model_dir = os.path.join(output_dir, f"run_{idx}")
    return best_params, best_loss, best_model_dir

# Function to compress and create a zip file
def create_zip_from_dir(output_dir, zip_name="fine_tuned_model.zip"):
    zip_path = os.path.join(output_dir, zip_name)
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(output_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, output_dir)
                zipf.write(file_path, arcname)
    return zip_path

# Streamlit UI
def main():
    st.title("Agentic AI Fine-Tuning Tool")
    st.write("Fine-tune Hugging Face models in 4 simple steps!")

    # Step 1: Model ID input
    st.header("Step 1: Provide Hugging Face Model ID")
    model_id = st.text_input("Enter the Hugging Face model ID (e.g., bert-base-uncased):", "bert-base-uncased")

    # Step 2: Upload dataset and handle it directly here
    st.header("Step 2: Upload Your Dataset")
    uploaded_file = st.file_uploader("Upload your dataset (CSV or JSON format):", type=["csv", "json"])
    file_format = st.selectbox("Select the file format of your dataset:", ["csv", "json"])
    input_column = st.text_input("Enter the column name containing text input:", "text")

    # Check if the user uploaded a file
    if uploaded_file is not None:
        # Handle CSV file upload
        if file_format == "csv":
            # Load the dataset into pandas DataFrame
            df = pd.read_csv(uploaded_file)
            st.write("CSV File Data:")
            st.dataframe(df)
            if input_column in df.columns:
                st.write(f"Displaying text data from the '{input_column}' column:")
                st.write(df[input_column].head())
            else:
                st.error(f"Column '{input_column}' not found in the dataset.")
        
        # Handle JSON file upload
        elif file_format == "json":
            try:
                json_data = json.loads(uploaded_file.getvalue())
                st.write("JSON File Data:")
                st.json(json_data)
                # Check if input_column exists in the first JSON object
                if isinstance(json_data, list) and len(json_data) > 0 and input_column in json_data[0]:
                    st.write(f"Displaying text data from the '{input_column}' field:")
                    st.write([item[input_column] for item in json_data[:5]])  # Show first 5 entries
                else:
                    st.error(f"Field '{input_column}' not found in the JSON data.")
            except json.JSONDecodeError:
                st.error("Failed to decode JSON content.")

    # Step 3: Parameter Grid for fine-tuning
    st.header("Step 3: Define Hyperparameters")
    learning_rates = st.text_input("Learning rates (comma-separated):", "5e-5,3e-5")
    batch_sizes = st.text_input("Batch sizes (comma-separated):", "16,32")
    epochs = st.text_input("Epochs (comma-separated):", "2,3")

    # Step 4: Start Fine-Tuning
    st.header("Step 4: Start Fine-Tuning")
    output_dir = st.text_input("Enter the output directory to save results:", "./results")
    start_button = st.button("Start Fine-Tuning")

    # When Start Button is clicked
    if start_button and uploaded_file:
        st.write("Loading model and tokenizer...")
        model, tokenizer = load_model_and_tokenizer(model_id)

        st.write("Loading dataset...")
        dataset = load_data(uploaded_file, file_format)
        dataset = format_data(dataset["train"], tokenizer, input_column)
        dataset = dataset.train_test_split(test_size=0.2)

        st.write("Starting fine-tuning...")
        param_grid = {
            "learning_rate": [float(x) for x in learning_rates.split(",")],
            "batch_size": [int(x) for x in batch_sizes.split(",")],
            "epochs": [int(x) for x in epochs.split(",")]
        }
        best_params, best_loss, best_model_dir = fine_tune(model, tokenizer, dataset, param_grid, output_dir)
        st.success(f"Fine-tuning completed! Best parameters: {best_params} with loss: {best_loss}")

        # Compress and provide download link
        st.write("Preparing model for download...")
        zip_path = create_zip_from_dir(best_model_dir)
        with open(zip_path, "rb") as file:
            st.download_button(
                label="Download Fine-Tuned Model",
                data=file,
                file_name="fine_tuned_model.zip",
                mime="application/zip"
            )

if __name__ == "__main__":
    main()

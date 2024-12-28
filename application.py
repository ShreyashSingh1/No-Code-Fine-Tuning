import streamlit as st
import pandas as pd
from src.pipeline.agents import ClassificationAgentTunner, LLMAgentTunner

def main():
    st.title("Agentic AI Fine-Tuning Tool")
    st.write("Fine-tune Hugging Face models in 4 simple steps!")

    # Step 1: Choose Fine-Tuning Type
    st.header("Step 1: Select Fine-Tuning Type")
    fine_tuning_type = st.selectbox("Choose the type of fine-tuning:", ["Sentiment Model Fine-Tuning", "LLM Fine-Tuning"])

    # Step 2: Model ID input
    st.header("Step 2: Provide Hugging Face Model ID")
    model_id = st.text_input("Enter the Hugging Face model ID (e.g., gpt2, albert-base-v2):", "gpt2" if fine_tuning_type == "LLM Fine-Tuning" else "albert-base-v2")

    # Step 3: Upload dataset
    st.header("Step 3: Upload Your Dataset")
    uploaded_file = st.file_uploader("Upload your dataset (CSV or JSON format):", type=["csv", "json"])
    file_format = st.selectbox("Select the file format of your dataset:", ["csv", "json"])

    input_column = st.text_input("Enter the column name containing text input:", "text")
    target_column = None

    if fine_tuning_type == "LLM Fine-Tuning":
        target_column = st.text_input("Enter the column name containing target output:", "target")

    if uploaded_file:
        # Display the uploaded dataset immediately
        if file_format == "csv":
            df = pd.read_csv(uploaded_file)
        elif file_format == "json":
            df = pd.read_json(uploaded_file)

        st.write("### Dataset Preview:")
        st.dataframe(df.head())

    # Step 4: Define hyperparameters
    st.header("Step 4: Define Hyperparameters")
    learning_rates = st.text_input("Learning rates (comma-separated):", "5e-5,3e-5")
    batch_sizes = st.text_input("Batch sizes (comma-separated):", "16,32")
    epochs = st.text_input("Epochs (comma-separated):", "2,3")

    # Step 5: Start fine-tuning
    st.header("Step 5: Start Fine-Tuning")
    output_dir = st.text_input("Enter the output directory to save results:", "./results")
    start_button = st.button("Start Fine-Tuning")

    if start_button and uploaded_file:
        st.write("Initializing FineTuner...")

        # Choose the appropriate fine-tuning class
        if fine_tuning_type == "Sentiment Model Fine-Tuning":
            fine_tuner = ClassificationAgentTunner(model_id, output_dir)
        else:
            fine_tuner = LLMAgentTunner(model_id, output_dir)

        st.write("Loading dataset...")
        dataset = fine_tuner.load_data(uploaded_file, file_format)

        # Format data for sentiment or LLM fine-tuning
        if fine_tuning_type == "Sentiment Model Fine-Tuning":
            dataset = fine_tuner.format_data(dataset["train"], input_column)
        else:
            dataset = fine_tuner.format_data(dataset["train"], input_column, target_column)

        dataset = dataset.train_test_split(test_size=0.2)

        st.write("Starting fine-tuning...")
        param_grid = {
            "learning_rate": [float(x) for x in learning_rates.split(",")],
            "batch_size": [int(x) for x in batch_sizes.split(",")],
            "epochs": [int(x) for x in epochs.split(",")]
        }

        with st.expander("Hyperparameters for Fine-Tuning"):
            param_df = pd.DataFrame(param_grid)
            st.dataframe(param_df)

        best_params, best_loss, best_model_dir = fine_tuner.fine_tune(dataset, param_grid)

        st.success(f"Fine-tuning completed! Best parameters: {best_params} with loss: {best_loss}")


        with st.expander("Fine-Tuning Results"):
            st.markdown(f"### Best Parameters:\n- **Learning Rate**: {best_params['learning_rate']}\n- **Batch Size**: {best_params['batch_size']}\n- **Epochs**: {best_params['epochs']}")
            st.markdown(f"### Best Loss: **{best_loss}**")

        # Add model training details
        with st.expander("Model Training Details"):
            st.write("The model is fine-tuned using the Hugging Face Transformers library. During training, the following steps are performed:")
            st.markdown("1. **Data Preparation**: The dataset is tokenized and split into training and validation sets.")
            st.markdown("2. **Model Initialization**: The selected pre-trained model is loaded from the Hugging Face Model Hub.")
            st.markdown("3. **Training Loop**: The model is trained using the specified hyperparameters (learning rate, batch size, and epochs).")
            st.markdown("4. **Validation**: The model is evaluated on the validation set to monitor performance and minimize loss.")
            st.markdown("5. **Checkpointing**: The best model weights are saved based on validation performance.")
        
        st.write("Preparing model for download...")

        zip_path = fine_tuner.create_zip_from_dir(best_model_dir)
        with open(zip_path, "rb") as file:
            st.download_button(
                label="Download Fine-Tuned Model",
                data=file,
                file_name="fine_tuned_model.zip",
                mime="application/zip"
            )

if __name__ == "__main__":
    main()

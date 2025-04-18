import streamlit as st
import pandas as pd
from src.pipeline.agents import ClassificationAgentTunner

def main():
    st.title("Agentic AI Fine-Tuning Tool")
    st.write("Fine-tune Hugging Face models in 4 simple steps!")

    # Step 1: Model ID input
    st.header("Step 1: Provide Hugging Face Model ID")
    model_id = st.text_input("Enter the Hugging Face model ID (e.g., albert-base-v2):", "albert-base-v2")

    # Step 2: Upload dataset and handle it directly here
    st.header("Step 2: Upload Your Dataset")
    uploaded_file = st.file_uploader("Upload your dataset (CSV or JSON format):", type=["csv", "json"])
    file_format = st.selectbox("Select the file format of your dataset:", ["csv", "json"])
    input_column = st.text_input("Enter the column name containing text input:", "text")
    
    if uploaded_file:
        # Display the uploaded dataset immediately
        if file_format == "csv":
            df = pd.read_csv(uploaded_file)
        elif file_format == "json":
            df = pd.read_json(uploaded_file)
    
        st.write("### Dataset Preview:")
        st.dataframe(df.head())

    # Step 3: Parameter Grid for fine-tuning
    st.header("Step 3: Define Hyperparameters")
    learning_rates = st.text_input("Learning rates (comma-separated):", "5e-5,3e-5")
    batch_sizes = st.text_input("Batch sizes (comma-separated):", "16,32")
    epochs = st.text_input("Epochs (comma-separated):", "2,3")

    # Step 4: Start Fine-Tuning
    st.header("Step 4: Start Fine-Tuning")
    output_dir = st.text_input("Enter the output directory to save results:", "./results")
    start_button = st.button("Start Fine-Tuning")


    if start_button and uploaded_file:
        st.write("Initializing FineTuner...")
        fine_tuner = ClassificationAgentTunner(model_id, output_dir)

        st.write("Loading dataset...")
        dataset = fine_tuner.load_data(uploaded_file, file_format)
        dataset = fine_tuner.format_data(dataset["train"], input_column)
        dataset = dataset.train_test_split(test_size=0.2)

        st.write("Starting fine-tuning...")
        param_grid = {
            "learning_rate": [float(x) for x in learning_rates.split(",")],
            "batch_size": [int(x) for x in batch_sizes.split(",")],
            "epochs": [int(x) for x in epochs.split(",")]
        }

        # Display parameters nicely
        with st.expander("Hyperparameters for Fine-Tuning"):
            param_df = pd.DataFrame(param_grid)
            st.dataframe(param_df)

        best_params, best_loss, best_model_dir = fine_tuner.fine_tune(dataset, param_grid)

        st.success(f"Fine-tuning completed! Best parameters: {best_params} with loss: {best_loss}")

        # Compress and provide download link
        st.write("Preparing model for download...")

        # Show results with better styling
        with st.expander("Fine-Tuning Results"):
            st.markdown(f"### Best Parameters:\n- **Learning Rate**: {best_params['learning_rate']}\n- **Batch Size**: {best_params['batch_size']}\n- **Epochs**: {best_params['epochs']}")
            st.markdown(f"### Best Loss: **{best_loss}**")

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

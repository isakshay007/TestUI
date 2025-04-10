import sys
import os
import pandas as pd

# Add ../src to import path so we can use BERT_Tagger
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from BERT import BERT_Tagger

if __name__ == "__main__":
    # Set up data and model paths
    current_dir = os.path.dirname(__file__)
    data_path = os.path.join(current_dir, "../data/stackoverflow_data.csv")
    save_path = os.path.join(current_dir, "bert_tagger.pkl")

    # Load dataset
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at: {data_path}")

    df = pd.read_csv(data_path)

    # Check expected columns
    required_columns = {"Title", "Question", "Tags"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_columns}")

    # Instantiate the tagger
    tagger = BERT_Tagger()

    # Train the model
    tagger.train(df, save_path=save_path)

    # Load the trained model and test prediction
    tagger.load_model(save_path)

    # Test input
    question = "How do I merge two dictionaries in a single expression in Python?"
    predicted_tags = tagger.predict(question, top_k=5)
    print("\nSample Question:")
    print(question)
    print("\nPredicted Tags:", predicted_tags)

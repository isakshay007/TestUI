import streamlit as st
import joblib
import os
from hmm import HMM_Tagger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def preprocess(text):
    return text.lower().strip()

def load_models():
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    ml_model = joblib.load(os.path.join(BASE_DIR, "models", "tagging_model.pkl"))
    mlb = joblib.load(os.path.join(BASE_DIR, "models", "tagging_mlb.pkl"))

    hmm_model = HMM_Tagger()
    hmm_model.load_model(os.path.join(BASE_DIR, "models", "hmm_model.pkl"))

    return ml_model, mlb, hmm_model

def predict_ml(model, mlb, title, description, threshold=0.08):
    combined_text = title + " " + description
    probs = model.predict_proba([combined_text])[0]
    prob_dict = dict(zip(mlb.classes_, probs))
    sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
    predicted_labels = [label for label, score in sorted_probs if score >= threshold]
    return predicted_labels, sorted_probs

def predict_hmm(hmm_model, title, description, threshold=0.1):
    combined_text = title + " " + description
    predicted_tags = hmm_model.predict(combined_text)

    input_sentence = preprocess(description)
    predicted_tags = list(set([preprocess(tag) for tag in predicted_tags]))

    all_text = [input_sentence] + predicted_tags
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_text)

    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    filtered_tags = [(tag, score) for tag, score in zip(predicted_tags, cosine_similarities) if score >= threshold]
    sorted_tags = sorted(filtered_tags, key=lambda x: x[1], reverse=True)
    return sorted_tags


# Set UI layout
st.set_page_config(page_title="StackOverflow Tag Generator", layout="wide")

# Title & Intro
st.title("🚀 StackOverflow Tag Generator")
st.markdown("""
Welcome to the **StackOverflow AI Tagging System**!  
This tool helps you generate relevant tags for your technical questions using either a **Logistic Regression-based ML model** or a **Hidden Markov Model (HMM)**.

👇 Start by selecting the model you'd like to use.
""")

# Load models
with st.spinner("Loading models..."):
    ml_model, mlb, hmm_model = load_models()

# Model Selection
model_choice = st.selectbox("📊 Select Tag Prediction Model", ["Logistic Regression (ML)", "Hidden Markov Model (HMM)"])
st.markdown("---")

# Title + Description inputs
st.subheader("📝 Provide your Question Details")
title = st.text_input("📌 Question Title", placeholder="e.g., How to merge dictionaries in Python?")
description = st.text_area("🧠 Question Description", placeholder="Provide details about your issue, approach, error, etc.", height=200)

# Submit button
if st.button("Generate Tags"):
    if not title.strip() or not description.strip():
        st.warning("Please fill in both title and description.")
    else:
        with st.spinner("Generating tags..."):
            if model_choice == "Logistic Regression (ML)":
                tags, scores = predict_ml(ml_model, mlb, title, description)
                st.subheader("🎯 Predicted Tags:")
                st.write(", ".join(tags) if tags else "No tags above threshold.")

                st.subheader("📊 Tag Probabilities:")
                for tag, score in scores[:10]:
                    st.write(f"**{tag}**: {score:.3f}")
            else:
                hmm_results = predict_hmm(hmm_model, title, description)
                st.subheader("🎯 Predicted Tags:")
                if hmm_results:
                    for tag, score in hmm_results[:10]:
                        st.write(f"**{tag}**: {score:.3f}")
                else:
                    st.write("No relevant tags found.")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit | Powered by Logistic Regression & HMM")
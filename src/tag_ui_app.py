import streamlit as st
import os
import joblib
import torch
import numpy as np
import pickle
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from hmm import HMM_Tagger
from torch import nn

# ========== MODEL DEFINITIONS ==========

class MiniTagTransformer(nn.Module):
    def __init__(self, num_tags):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_tags)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_output)

# ========== HELPERS ==========

@st.cache_resource

def load_models():
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    model_paths = {
        "ml_model": os.path.join(BASE_DIR, "models", "tagging_model.pkl"),
        "mlb_ml": os.path.join(BASE_DIR, "models", "tagging_mlb.pkl"),
        "bert_model": os.path.join(BASE_DIR, "models", "trained_model.pt"),
        "mlb_bert": os.path.join(BASE_DIR, "models", "mlb.pkl"),
        "hmm_model": os.path.join(BASE_DIR, "models", "hmm_model.pkl")
    }

    ml_model = joblib.load(model_paths["ml_model"])
    mlb_ml = joblib.load(model_paths["mlb_ml"])

    with open(model_paths["mlb_bert"], "rb") as f:
        mlb_bert = pickle.load(f)

    bert_model = MiniTagTransformer(num_tags=len(mlb_bert.classes_))
    bert_model.load_state_dict(torch.load(model_paths["bert_model"], map_location=torch.device("cpu")))
    bert_model.eval()

    hmm_model = HMM_Tagger()
    hmm_model.load_model(model_paths["hmm_model"])

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    return ml_model, mlb_ml, hmm_model, bert_model, mlb_bert, tokenizer


def preprocess(text):
    return text.lower().strip()


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


def predict_bert(text, model, tokenizer, mlb, threshold=0.05, show_top_k=5, fallback=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    tokenizer_inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    tokenizer_inputs = {key: val.to(device) for key, val in tokenizer_inputs.items()}

    with torch.no_grad():
        logits = model(tokenizer_inputs["input_ids"], tokenizer_inputs["attention_mask"])
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()

    top_probs = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)[:show_top_k]
    predicted_indices = np.where(probs >= threshold)[0]
    tags = [mlb.classes_[i] for i in predicted_indices]

    if fallback and not tags:
        tags = [mlb.classes_[i] for i, _ in top_probs]

    top_tags = [(mlb.classes_[i], p) for i, p in top_probs]
    return tags, top_tags

# ========== STREAMLIT UI ==========

st.set_page_config(page_title="StackOverflow Tag Generator", layout="wide")
st.title("🚀 StackOverflow Tag Generator")

st.markdown("""
This tool generates relevant tags for your technical questions using:
- Logistic Regression (TF-IDF)
- Hidden Markov Model (HMM)
- DistilBERT Transformer
""")

with st.spinner("🔄 Loading models..."):
    ml_model, mlb_ml, hmm_model, bert_model, mlb_bert, tokenizer = load_models()

model_choice = st.selectbox("📊 Select Model", ["Logistic Regression (ML)", "Hidden Markov Model (HMM)", "DistilBERT Transformer"])

st.subheader("📝 Provide Question Details")
title = st.text_input("📌 Question Title")
description = st.text_area("🧠 Question Description", height=200)

if st.button("Generate Tags"):
    if not title.strip() or not description.strip():
        st.warning("Please fill in both title and description")
    else:
        with st.spinner("⚙️ Generating tags..."):
            if model_choice == "Logistic Regression (ML)":
                tags, scores = predict_ml(ml_model, mlb_ml, title, description)
                st.subheader("🎯 Predicted Tags")
                st.write(", ".join(tags) if tags else "No tags above threshold")

                st.subheader("📊 Top Probabilities")
                for tag, score in scores[:10]:
                    st.write(f"**{tag}**: {score:.3f}")

            elif model_choice == "Hidden Markov Model (HMM)":
                hmm_results = predict_hmm(hmm_model, title, description)
                st.subheader("🎯 Predicted Tags")
                for tag, score in hmm_results[:10]:
                    st.write(f"**{tag}**: {score:.3f}")

            else:
                combined = title + " " + description
                tags, scores = predict_bert(combined, bert_model, tokenizer, mlb_bert)
                st.subheader("🎯 Predicted Tags")
                st.write(", ".join(tags))
                st.subheader("📊 Top Tag Probabilities")
                for tag, prob in scores:
                    st.write(f"**{tag}**: {prob:.3f}")

st.markdown("---")
st.caption("Made with ❤️ using Streamlit | Powered by ML, HMM, and BERT")
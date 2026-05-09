# app.py - Clean Ensemble Sentiment Analyzer
import streamlit as st
import joblib
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import json
import matplotlib.pyplot as plt

# NLTK setup
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text_nltk(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+|#', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return ' '.join(words)

@st.cache_resource
def load_models():
    lr_model = joblib.load("logistic_regression_model.pkl")
    bert_model = DistilBertForSequenceClassification.from_pretrained("./saved_bert_model")
    bert_tokenizer = DistilBertTokenizerFast.from_pretrained("./saved_bert_model")
    with open("./saved_bert_model/label_map.json", "r") as f:
        label_map = json.load(f)
    id2label = {int(k): v for k, v in label_map.items()}
    return lr_model, bert_model, bert_tokenizer, id2label

lr_model, bert_model, bert_tokenizer, id2label = load_models()

# ---------- UI ----------
st.set_page_config(page_title="Sentiment Analyzer", layout="centered")
st.title("📘 Facebook Post Sentiment Analyzer")
st.markdown("Paste a post and see the ensemble sentiment from both models.")

user_input = st.text_area("Enter post text:", height=150)

if st.button("Analyze Sentiment"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Analyzing..."):
            # Logistic Regression
            cleaned = clean_text_nltk(user_input)
            lr_proba = lr_model.predict_proba([cleaned])[0]
            lr_classes = lr_model.classes_
            
            # DistilBERT
            inputs = bert_tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = bert_model(**inputs)
            bert_probs = torch.softmax(outputs.logits, dim=1).numpy()[0]
            
            # Ensemble average (class order: negative, neutral, positive)
            class_order = ['negative', 'neutral', 'positive']
            
            # Map probabilities into unified order
            lr_dict = {cls: prob for cls, prob in zip(lr_classes, lr_proba)}
            bert_dict = {id2label[i]: prob for i, prob in enumerate(bert_probs)}
            
            ensemble_probs = {}
            for cls in class_order:
                ensemble_probs[cls] = (lr_dict.get(cls, 0) + bert_dict.get(cls, 0)) / 2.0
            
            final_pred = max(ensemble_probs, key=ensemble_probs.get)

        # Display ensemble result
        emoji = {"positive": "😊", "neutral": "😐", "negative": "😠"}
        st.markdown(f"### Final Sentiment: {emoji[final_pred]} **{final_pred.capitalize()}**")
        st.write("**Confidence scores:**")
        st.json({cls: f"{prob:.1%}" for cls, prob in ensemble_probs.items()})
        st.bar_chart(ensemble_probs)

        # Optional: show individual predictions in a small expander
        with st.expander("See individual model predictions"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Logistic Regression**")
                st.write({cls: f"{prob:.1%}" for cls, prob in zip(lr_classes, lr_proba)})
            with col2:
                st.write("**DistilBERT**")
                st.write({id2label[i]: f"{prob:.1%}" for i, prob in enumerate(bert_probs)})

st.markdown("---")
st.caption("Ensemble of TF‑IDF + Logistic Regression and DistilBERT")
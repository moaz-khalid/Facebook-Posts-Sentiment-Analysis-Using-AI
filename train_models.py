# train_models.py
# Full AI pipeline for Facebook Post Sentiment Analysis

import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import joblib
import torch
from transformers import (DistilBertTokenizerFast,
                          DistilBertForSequenceClassification,
                          Trainer, TrainingArguments)
from datasets import Dataset

# ======================================================
# 1. DATASET LOADING
# ======================================================
print("Loading dataset...")
df = pd.read_csv("sentiment_dataset.csv")
df = df.rename(columns={"text": "post_text", "SENTIMENT": "sentiment"})
df = df[["post_text", "sentiment"]]
print("Dataset shape:", df.shape)

# ======================================================
# 2. DATASET EXPLORATION
# ======================================================
print("\n--- Exploration ---")
print(df.info())
print("\nMissing values:\n", df.isnull().sum())
print("\nSentiment distribution:\n", df['sentiment'].value_counts())

# Drop any missing rows (if any)
df.dropna(subset=['post_text', 'sentiment'], inplace=True)

# ======================================================
# 3. DATASET VISUALIZATION
# ======================================================
print("\nGenerating visualizations...")
sns.set_style('whitegrid')

# Bar plot of sentiment counts
plt.figure(figsize=(6,4))
sns.countplot(x='sentiment', data=df, palette='viridis')
plt.title("Sentiment Distribution")
plt.tight_layout()
plt.savefig("sentiment_distribution.png")
plt.close()

# Word clouds for each sentiment
for sentiment in df['sentiment'].unique():
    text = ' '.join(df[df['sentiment'] == sentiment]['post_text'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f"Word Cloud - {sentiment}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"wordcloud_{sentiment}.png")
    plt.close()

print("Visualizations saved as PNG files.")

# ======================================================
# 4. DATASET PREPROCESSING (Two versions)
# ======================================================

# --- NLTK based preprocessing for classical ML ---
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text_nltk(text):
    # lowercase
    text = text.lower()
    # remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    # remove mentions and hashtags
    text = re.sub(r'@\w+|#', '', text)
    # keep only letters and spaces
    text = re.sub(r'[^a-z\s]', '', text)
    # tokenize and remove stopwords + lemmatize
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return ' '.join(words)

print("Preprocessing text for classical model...")
df['clean_text'] = df['post_text'].apply(clean_text_nltk)

# For the transformer, we keep the original text (post_text)

# ======================================================
# 5. MODEL DEVELOPMENT / BUILDING
# ======================================================

# --- A. Classical ML: TF-IDF + Logistic Regression Pipeline ---
print("\nBuilding TF-IDF + Logistic Regression pipeline...")
pipeline_lr = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
    ('clf', LogisticRegression(max_iter=1000, random_state=42))
])

# --- B. Transformer: DistilBERT fine-tuning ---
print("Loading DistilBERT tokenizer and model...")
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
model_bert = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased', num_labels=3
)

# Convert sentiment labels to integers
label2id = {'negative': 0, 'neutral': 1, 'positive': 2}
id2label = {v: k for k, v in label2id.items()}
df['label'] = df['sentiment'].map(label2id)

# ======================================================
# 6. TRAIN / TEST SPLIT
# ======================================================
# Stratified split to keep class proportions
X_train, X_temp, y_train, y_temp = train_test_split(
    df['clean_text'], df['sentiment'], test_size=0.3,
    stratify=df['sentiment'], random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5,
    stratify=y_temp, random_state=42
)

# For transformer, we need the original text and integer labels
train_texts = df.loc[X_train.index, 'post_text']
val_texts   = df.loc[X_val.index, 'post_text']
test_texts  = df.loc[X_test.index, 'post_text']
train_labels = df.loc[X_train.index, 'label']
val_labels   = df.loc[X_val.index, 'label']
test_labels  = df.loc[X_test.index, 'label']

print(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")

# ======================================================
# 6 (cont.) MODEL TRAINING - Classical
# ======================================================
print("\nTraining Logistic Regression model...")
pipeline_lr.fit(X_train, y_train)

# ======================================================
# 7. MODEL TESTING - Classical
# ======================================================
print("\n--- Classical Model Evaluation ---")
y_pred_lr = pipeline_lr.predict(X_test)
print(classification_report(y_test, y_pred_lr))
cm = confusion_matrix(y_test, y_pred_lr, labels=['negative', 'neutral', 'positive'])
sns.heatmap(cm, annot=True, fmt='d', xticklabels=['neg','neu','pos'], yticklabels=['neg','neu','pos'])
plt.title("Confusion Matrix - Logistic Regression")
plt.tight_layout()
plt.savefig("confusion_matrix_lr.png")
plt.close()

# ======================================================
# 7 (cont.) HYPERPARAMETER TUNING - Classical
# ======================================================
print("\nTuning Logistic Regression hyperparameters...")
param_grid = {
    'tfidf__max_features': [3000, 5000],
    'tfidf__ngram_range': [(1,1), (1,2)],
    'clf__C': [0.1, 1, 10]
}
grid = GridSearchCV(pipeline_lr, param_grid, cv=3, scoring='f1_macro', n_jobs=-1)
grid.fit(X_train, y_train)
print("Best parameters:", grid.best_params_)
print("Best CV F1-macro:", grid.best_score_)

# Use best model
pipeline_lr = grid.best_estimator_

# Re-evaluate on test set
y_pred_best = pipeline_lr.predict(X_test)
print("\nTuned model test performance:")
print(classification_report(y_test, y_pred_best))

# Save the classical model
joblib.dump(pipeline_lr, "logistic_regression_model.pkl")
print("Classical model saved as logistic_regression_model.pkl")

# ======================================================
# 8. TRAIN TRANSFORMER (DistilBERT)
# ======================================================
print("\n--- Preparing Transformer dataset ---")

# Create Hugging Face Dataset objects
train_dataset = Dataset.from_dict({
    'text': train_texts.tolist(),
    'label': train_labels.tolist()
})
val_dataset = Dataset.from_dict({
    'text': val_texts.tolist(),
    'label': val_labels.tolist()
})
test_dataset = Dataset.from_dict({
    'text': test_texts.tolist(),
    'label': test_labels.tolist()
})

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)

print("Tokenizing datasets...")
train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset   = val_dataset.map(tokenize_function, batched=True)
test_dataset  = test_dataset.map(tokenize_function, batched=True)

# Set format to PyTorch tensors
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Training arguments (small epochs for mini-project, tune as needed)
training_args = TrainingArguments(
    output_dir='./bert_results',
    num_train_epochs=3,                  # you can increase to 4-5 if GPU available
    per_device_train_batch_size=8,       # reduce if out of memory
    per_device_eval_batch_size=8,
    eval_strategy='epoch', 
    save_strategy='epoch',
    logging_dir='./logs',
    load_best_model_at_end=True,
    metric_for_best_model='eval_f1_macro',
    logging_steps=10,
    learning_rate=2e-5,
    weight_decay=0.01,
    save_total_limit=1,
)

# Compute metrics function
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average='macro')
    return {'accuracy': acc, 'f1_macro': f1_macro}

# Trainer
trainer = Trainer(
    model=model_bert,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

print("Fine-tuning DistilBERT (this may take several minutes)...")
trainer.train()

# Evaluate on test set
print("\n--- Transformer Evaluation ---")
test_results = trainer.predict(test_dataset)
preds_bert = np.argmax(test_results.predictions, axis=1)
print(classification_report(test_labels, preds_bert, target_names=['negative', 'neutral', 'positive']))

# Save the transformer model and tokenizer
model_bert.save_pretrained("./saved_bert_model")
tokenizer.save_pretrained("./saved_bert_model")
print("Transformer model and tokenizer saved in './saved_bert_model'")

# Also save id2label mapping
import json
with open("./saved_bert_model/label_map.json", "w") as f:
    json.dump({str(k): v for k, v in id2label.items()}, f)

print("\nAll models trained and saved successfully!")
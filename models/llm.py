"""
POS Tagger using Hugging Face (local saved copy)
------------------------------------------------

Requirements:
    pip install transformers torch nltk

This script:
  • Loads a pretrained POS tagging model from Hugging Face (once)
  • Saves a local copy for future use
  • Loads from the local copy for inference
  • Runs tagging on input text
"""

import os
import nltk
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Ensure NLTK punkt is available
nltk.download("punkt", quiet=True)

# Hugging Face model name
MODEL_NAME = "vblagoje/bert-english-uncased-finetuned-pos"

# Directory to save local copy of model
LOCAL_MODEL_DIR = "saved_pos_model"
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

# -------------------------
# Load model + tokenizer
# -------------------------
if os.listdir(LOCAL_MODEL_DIR):
    # If local copy exists, load from there
    print("Loading model from local directory...")
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR)
    model = AutoModelForTokenClassification.from_pretrained(LOCAL_MODEL_DIR)
else:
    # Download from Hugging Face and save locally
    print("Downloading model from Hugging Face...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
    tokenizer.save_pretrained(LOCAL_MODEL_DIR)
    model.save_pretrained(LOCAL_MODEL_DIR)
    print(f"Model saved locally at '{LOCAL_MODEL_DIR}'")

# -------------------------
# Create POS tagging pipeline
# -------------------------
pos_tagger = pipeline(
    "token-classification",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple"
)

# -------------------------
# Demo inference
# -------------------------
text = "The quick brown fox jumps over the lazy dog."
results = pos_tagger(text)

print("\nPOS tagging results:\n")
for item in results:
    print(f"{item['word']:<15} → {item['entity_group']}")



# """
# Hugging Face POS Tagger - Local Inference Script
# -------------------------------------------------

# Requirements:
#     pip install transformers torch nltk

# This script:
#   • Loads a locally saved POS tagging model
#   • Runs inference on a given sentence
# """

# import nltk
# from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
# import os

# # Ensure NLTK punkt tokenizer is available
# nltk.download("punkt", quiet=True)

# # Path to the locally saved model
# LOCAL_MODEL_DIR = "saved_pos_model"  # Make sure this path points to the saved model folder

# if not os.path.exists(LOCAL_MODEL_DIR) or not os.listdir(LOCAL_MODEL_DIR):
#     raise FileNotFoundError(f"No saved model found at '{LOCAL_MODEL_DIR}'. Please save the model first.")

# # Load model and tokenizer from local folder
# tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR)
# model = AutoModelForTokenClassification.from_pretrained(LOCAL_MODEL_DIR)

# # Create a Hugging Face pipeline for token classification
# pos_tagger = pipeline(
#     "token-classification",
#     model=model,
#     tokenizer=tokenizer,
#     aggregation_strategy="simple"
# )

# # -------------------------
# # Function for POS tagging
# # -------------------------
# def tag_sentence(text: str):
#     results = pos_tagger(text)
#     tagged = [(item['word'], item['entity_group']) for item in results]
#     return tagged

# # -------------------------
# # Demo usage
# # -------------------------
# if __name__ == "__main__":
#     sentence = "The quick brown fox jumps over the lazy dog."
#     tags = tag_sentence(sentence)
#     print("\nPOS tagging results:\n")
#     for word, tag in tags:
#         print(f"{word:<15} → {tag}")

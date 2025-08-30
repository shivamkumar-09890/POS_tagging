import os
import json
import pickle

BASE_DIR = "/home/shivam/cs772/Assignment1"
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

def load_processed_data():
    """Load datasets + vocab from preprocessing step."""
    with open(os.path.join(PROCESSED_DIR, "train.pkl"), "rb") as f:
        X_train, y_train = pickle.load(f)
    with open(os.path.join(PROCESSED_DIR, "val.pkl"), "rb") as f:
        X_val, y_val = pickle.load(f)
    with open(os.path.join(PROCESSED_DIR, "test.pkl"), "rb") as f:
        X_test, y_test = pickle.load(f)

    with open(os.path.join(PROCESSED_DIR, "word2idx.json"), "r") as f:
        word2idx = json.load(f)
    with open(os.path.join(PROCESSED_DIR, "tag2idx.json"), "r") as f:
        tag2idx = json.load(f)
    with open(os.path.join(PROCESSED_DIR, "idx2word.json"), "r") as f:
        idx2word = json.load(f)
    with open(os.path.join(PROCESSED_DIR, "idx2tag.json"), "r") as f:
        idx2tag = json.load(f)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), word2idx, tag2idx, idx2word, idx2tag

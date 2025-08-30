import os
import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from models.hmm import HMMPOSTagger
from models.utils.dataLoder import load_processed_data
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

BASE_DIR = "/home/shivam/cs772/Assignment1"
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models", "saved_hmm_pos")
LOG_DIR = os.path.join(BASE_DIR, "models","results", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
OUT_PATH = os.path.join(LOG_DIR, "hmm_metrics.json")

def plot_confusion_matrix(y_true, y_pred, labels, save_path= "/home/shivam/cs772/Assignment1/models/results/plots/confusion_hmm_matrix.png"):
    """
    Plots the confusion matrix using seaborn heatmap.

    Args:
        y_true (list): True labels.
        y_pred (list): Predicted labels.
        labels (list): List of all possible labels in the dataset.
        save_path (str, optional): Path to save the plot image. If None, plot is shown.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"✅ Confusion matrix saved to {save_path}")
    else:
        plt.show()

def evaluate_hmm():
    # Load processed data
    (_, _), (_, _), (X_test, y_test), word2idx, tag2idx, idx2word, idx2tag = load_processed_data()

    # Load saved model
    hmm = HMMPOSTagger.load(MODEL_DIR)

    y_true, y_pred = [], []

    for words, tags in zip(X_test, y_test):
        # Reconstruct sentence words (skip PAD=0)
        sent_words = [idx2word[str(w)] for w in words if w != 0]
        gold_tags = [idx2tag[str(t)] for w, t in zip(words, tags) if w != 0]

        # Predict
        pred_tags = hmm.predict(sent_words)

        y_true.extend(gold_tags)
        y_pred.extend(pred_tags)

    # Compute overall metrics
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted"
    )

    # Compute per-tag accuracy
    per_tag_acc = {}
    labels = list(idx2tag.values())
    for label in labels:
        indices = [i for i, t in enumerate(y_true) if t == label]
        if indices:
            correct = sum(1 for i in indices if y_true[i] == y_pred[i])
            per_tag_acc[label] = correct / len(indices)
        else:
            per_tag_acc[label] = None  # in case tag not present in test set

    results = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "per_tag_accuracy": per_tag_acc,
        "classification_report": classification_report(y_true, y_pred, output_dict=True),
    }

    # Save results
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=4)

    print(f"✅ HMM metrics saved to {OUT_PATH}")

    plot_confusion_matrix(y_true, y_pred, labels=labels)

if __name__ == "__main__":
    evaluate_hmm()

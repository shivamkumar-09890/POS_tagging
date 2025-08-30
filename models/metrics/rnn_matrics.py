import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from models.rnn import BiLSTMTagger, IndexedPOSDataset, collate_batch, load_vocabs, TAG_PAD, DEVICE
from models.utils.dataLoder import load_processed_data
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
# Paths
BASE_DIR = "/home/shivam/cs772/Assignment1"
MODEL_DIR = os.path.join(BASE_DIR, "models", "saved_rnn_pos")
LOG_DIR = os.path.join(BASE_DIR,"models","results", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
OUT_PATH = os.path.join(LOG_DIR, "rnn_metrics.json")
DATA_DIR = os.path.join(BASE_DIR,"data","processed")

def plot_confusion_matrix(y_true, y_pred, labels, save_path="/home/shivam/cs772/Assignment1/models/results/plots/confusion_rnn_matrix_per.png"):
    """
    Plots the confusion matrix using seaborn heatmap.

    Args:
        y_true (list): True labels.
        y_pred (list): Predicted labels.
        labels (list): List of all possible labels in the dataset.
        save_path (str, optional): Path to save the plot image. If None, plot is shown.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"✅ Confusion matrix saved to {save_path}")
    else:
        plt.show()

def evaluate_rnn():
    # 🔹 Load processed data
    (_, _), (_, _), (X_test, y_test), _, _, _, _ = load_processed_data()

    # 🔹 Load vocabs
    word2idx, tag2idx, idx2word, idx2tag = load_vocabs(DATA_DIR)

    vocab_size = len(word2idx)
    tagset_size = len(tag2idx)

    # 🔹 Build test dataset/loader
    test_ds = IndexedPOSDataset(X_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=collate_batch)

    # 🔹 Load model
    model = BiLSTMTagger(vocab_size=vocab_size, tagset_size=tagset_size).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "model.pt"), map_location=DEVICE))
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for x_batch, y_batch, lengths in test_loader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            lengths = lengths.to(DEVICE)

            logits = model(x_batch, lengths)   # (B, L, C)
            preds = logits.argmax(dim=-1)      # (B, L)

            mask = (y_batch != TAG_PAD)

            for gold_seq, pred_seq, mask_seq in zip(y_batch, preds, mask):
                for g, p, m in zip(gold_seq.tolist(), pred_seq.tolist(), mask_seq.tolist()):
                    if m:   # ignore PAD positions
                        y_true.append(idx2tag[str(g)])
                        y_pred.append(idx2tag[str(p)])

    # 🔹 Compute metrics
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted"
    )

    results = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "classification_report": classification_report(y_true, y_pred, output_dict=True),
    }

    # 🔹 Save metrics
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=4)

    print(f"✅ RNN metrics saved to {OUT_PATH}")
    labels = list(idx2tag.values())
    plot_confusion_matrix(y_true, y_pred, labels=labels)

if __name__ == "__main__":
    evaluate_rnn()

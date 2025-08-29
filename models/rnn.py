# rnn_pos_tagger.py
import os
import json
import random
import pickle
from typing import List, Tuple, Dict

import nltk
from nltk.corpus import brown
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# -----------------------------
# Config / Hyperparameters
# -----------------------------
nltk.download("brown", quiet=True)
nltk.download("universal_tagset", quiet=True)

MODEL_DIR = "saved_rnn_pos"
os.makedirs(MODEL_DIR, exist_ok=True)

EMBED_DIM = 128
HIDDEN_DIM = 256
NUM_LAYERS = 1
BIDIRECTIONAL = True
BATCH_SIZE = 64
EPOCHS = 8
LR = 0.001
MAX_VOCAB = 50000   # keep large enough
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
TAG_PAD = -100      # ignore index for loss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Helpers: build vocab, encode
# -----------------------------
def build_vocab(sentences: List[List[Tuple[str, str]]]):
    words = set()
    tags = set()
    for sent in sentences:
        for w, t in sent:
            words.add(w.lower())
            tags.add(t)
    # Reserve indices: 0 -> PAD, 1 -> UNK
    sorted_words = sorted(list(words))[:MAX_VOCAB]
    word2idx = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for i, w in enumerate(sorted_words, start=2):
        word2idx[w] = i
    tag2idx = {tag: i for i, tag in enumerate(sorted(tags))}
    idx2word = {i: w for w, i in word2idx.items()}
    idx2tag = {i: t for t, i in tag2idx.items()}
    return word2idx, tag2idx, idx2word, idx2tag

def encode_sentence(sent: List[Tuple[str, str]], word2idx: Dict[str,int], tag2idx: Dict[str,int]):
    words = [w.lower() for w, _ in sent]
    tags = [t for _, t in sent]
    x = [word2idx.get(w, word2idx[UNK_TOKEN]) for w in words]
    y = [tag2idx[t] for t in tags]
    return x, y

# -----------------------------
# Dataset + collate_fn
# -----------------------------
class POSDataset(Dataset):
    def __init__(self, tagged_sents: List[List[Tuple[str,str]]], word2idx, tag2idx):
        self.data = [encode_sentence(s, word2idx, tag2idx) for s in tagged_sents]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]   # (x_list, y_list)

def collate_batch(batch):
    # batch: list of (x, y) where x,y are lists
    xs, ys = zip(*batch)
    lengths = [len(x) for x in xs]
    max_len = max(lengths)
    padded_x = []
    padded_y = []
    for x, y in zip(xs, ys):
        pad_len = max_len - len(x)
        padded_x.append(x + [0]*pad_len)  # PAD index = 0
        padded_y.append(y + [TAG_PAD]*pad_len)  # label ignore pad
    return (torch.tensor(padded_x, dtype=torch.long),
            torch.tensor(padded_y, dtype=torch.long),
            torch.tensor(lengths, dtype=torch.long))

# -----------------------------
# Model: Embedding + BiLSTM + Linear
# -----------------------------
class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size, tagset_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM,
                 num_layers=NUM_LAYERS, bidirectional=BIDIRECTIONAL, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim,
                            hidden_dim,
                            num_layers=num_layers,
                            bidirectional=bidirectional,
                            batch_first=True,
                            dropout=dropout if num_layers>1 else 0.0)
        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Linear(lstm_out_dim, tagset_size)

    def forward(self, x, lengths):
        # x: (batch, seq_len)
        emb = self.embedding(x)  # (batch, seq_len, emb)
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)  # (batch, seq_len, hidden*dirs)
        logits = self.fc(out)  # (batch, seq_len, tagset)
        return logits

# -----------------------------
# Training / Evaluation loops
# -----------------------------
def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for x_batch, y_batch, lengths in dataloader:
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        lengths = lengths.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x_batch, lengths)  # (B, L, C)
        # reshape for loss: (B*L, C) and targets (B*L)
        B, L, C = logits.shape
        logits_flat = logits.view(B*L, C)
        targets_flat = y_batch.view(B*L)
        loss = criterion(logits_flat, targets_flat)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, idx2tag):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for x_batch, y_batch, lengths in dataloader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            lengths = lengths.to(DEVICE)
            logits = model(x_batch, lengths)  # (B, L, C)
            preds = logits.argmax(dim=-1)     # (B, L)
            # count ignoring tag pad (TAG_PAD)
            mask = (y_batch != TAG_PAD).to(torch.long)
            total += mask.sum().item()
            correct += ((preds == y_batch).to(torch.long) * mask).sum().item()
    acc = correct / total if total else 0.0
    return acc

# -----------------------------
# Save / Load utilities
# -----------------------------
def save_checkpoint(model, word2idx, tag2idx, idx2word, idx2tag, path=MODEL_DIR):
    torch.save(model.state_dict(), os.path.join(path, "model.pt"))
    with open(os.path.join(path, "word2idx.json"), "w") as f:
        json.dump(word2idx, f)
    with open(os.path.join(path, "tag2idx.json"), "w") as f:
        json.dump(tag2idx, f)
    with open(os.path.join(path, "idx2word.json"), "w") as f:
        json.dump(idx2word, f)
    with open(os.path.join(path, "idx2tag.json"), "w") as f:
        json.dump(idx2tag, f)

def load_vocabs(path=MODEL_DIR):
    with open(os.path.join(path, "word2idx.json")) as f:
        word2idx = json.load(f)
    with open(os.path.join(path, "tag2idx.json")) as f:
        tag2idx = json.load(f)
    with open(os.path.join(path, "idx2word.json")) as f:
        idx2word = json.load(f)
    with open(os.path.join(path, "idx2tag.json")) as f:
        idx2tag = json.load(f)
    # keys in JSON are strings — convert tag2idx values to int keys already are fine
    return word2idx, tag2idx, idx2word, idx2tag

# -----------------------------
# Main pipeline
# -----------------------------
def main():
    # load dataset
    tagged = list(brown.tagged_sents(tagset="universal"))
    # optionally shuffle then split
    random.seed(42)
    random.shuffle(tagged)
    train_sents, test_sents = train_test_split(tagged, test_size=0.2, random_state=42)
    train_sents, val_sents = train_test_split(train_sents, test_size=0.1, random_state=42)

    print(f"Train: {len(train_sents)}, Val: {len(val_sents)}, Test: {len(test_sents)}")

    word2idx, tag2idx, idx2word, idx2tag = build_vocab(train_sents)
    vocab_size = len(word2idx)
    tagset_size = len(tag2idx)

    print("Vocab size:", vocab_size, "Tagset size:", tagset_size)

    train_ds = POSDataset(train_sents, word2idx, tag2idx)
    val_ds = POSDataset(val_sents, word2idx, tag2idx)
    test_ds = POSDataset(test_sents, word2idx, tag2idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

    # model, optimizer, criterion
    model = BiLSTMTagger(vocab_size=vocab_size, tagset_size=tagset_size).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=TAG_PAD)

    best_val = 0.0
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_acc = evaluate(model, val_loader, idx2tag)
        print(f"Epoch {epoch:2d} | Train loss: {train_loss:.4f} | Val acc: {val_acc:.4f}")
        # save best model
        if val_acc > best_val:
            best_val = val_acc
            save_checkpoint(model, word2idx, tag2idx, idx2word, idx2tag)
            print(f"  Saved best model (val acc {best_val:.4f})")

    # final test
    print("Loading best model for test evaluation...")
    # load weights
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "model.pt"), map_location=DEVICE))
    test_acc = evaluate(model, test_loader, idx2tag)
    print(f"Test token-level accuracy: {test_acc:.4f}")

    # demo inference on a single sentence
    demo = "The quick brown fox jumps over the lazy dog .".split()
    # encode demo
    word2idx_local, tag2idx_local, idx2word_local, idx2tag_local = word2idx, tag2idx, idx2word, idx2tag
    x_demo = [word2idx_local.get(w.lower(), word2idx_local[UNK_TOKEN]) for w in demo]
    lengths_demo = torch.tensor([len(x_demo)], dtype=torch.long)
    x_demo_tensor = torch.tensor([x_demo], dtype=torch.long).to(DEVICE)
    model.eval()
    with torch.no_grad():
        logits = model(x_demo_tensor, lengths_demo.to(DEVICE))
        preds = logits.argmax(dim=-1).squeeze(0).cpu().tolist()
    print("\nDemo tagging:")
    for tok, p in zip(demo, preds):
        print(f"{tok:12} -> {idx2tag_local[str(p)] if isinstance(list(idx2tag_local.keys())[0], str) else idx2tag_local[p] }")
    # note: idx2tag loaded from build_vocab uses int keys; saved idx2tag JSON keys are strings if reloaded.

if __name__ == "__main__":
    main()

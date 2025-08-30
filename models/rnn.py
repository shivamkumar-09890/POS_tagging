# rnn_pos_tagger.py
import os
import json
from typing import List, Tuple, Dict
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from models.utils.dataLoder import load_processed_data

MODEL_DIR = "models/saved_rnn_pos"
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

class IndexedPOSDataset(Dataset):
    """Dataset for already indexed sequences (X, y)."""
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def collate_batch(batch):
    xs, ys = zip(*batch)
    lengths = [len(x) for x in xs]
    max_len = max(lengths)
    padded_x = []
    padded_y = []
    for x, y in zip(xs, ys):
        pad_len = max_len - len(x)
        padded_x.append(x + [0]*pad_len)
        padded_y.append(y + [TAG_PAD]*pad_len)
    return (torch.tensor(padded_x, dtype=torch.long),
            torch.tensor(padded_y, dtype=torch.long),
            torch.tensor(lengths, dtype=torch.long))

class LSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Combine all gates into one big linear transform for efficiency
        self.x2h = nn.Linear(input_dim, 4 * hidden_dim)
        self.h2h = nn.Linear(hidden_dim, 4 * hidden_dim)

    def forward(self, x_t, h_prev, c_prev):
        # x_t: (batch, input_dim)
        # h_prev, c_prev: (batch, hidden_dim)
        gates = self.x2h(x_t) + self.h2h(h_prev)
        i, f, g, o = gates.chunk(4, dim=-1)

        i = torch.sigmoid(i)          # input gate
        f = torch.sigmoid(f)          # forget gate
        g = torch.tanh(g)             # candidate cell
        o = torch.sigmoid(o)          # output gate

        c_t = f * c_prev + i * g
        h_t = o * torch.tanh(c_t)
        return h_t, c_t

class LSTMLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.cell = LSTMCell(input_dim, hidden_dim)

    def forward(self, x, lengths):
        # x: (batch, seq_len, input_dim)
        batch_size, seq_len, _ = x.size()
        h = torch.zeros(batch_size, self.cell.hidden_dim, device=x.device)
        c = torch.zeros(batch_size, self.cell.hidden_dim, device=x.device)

        outputs = []
        for t in range(seq_len):
            h, c = self.cell(x[:, t, :], h, c)
            outputs.append(h.unsqueeze(1))
        outputs = torch.cat(outputs, dim=1)  # (batch, seq_len, hidden_dim)
        return outputs

class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size, tagset_size,
                 embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM,
                 num_layers=NUM_LAYERS, bidirectional=True, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim

        # Forward layers
        self.layers_fwd = nn.ModuleList([
            LSTMLayer(embed_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])

        if bidirectional:
            # Backward layers
            self.layers_bwd = nn.ModuleList([
                LSTMLayer(embed_dim if i == 0 else hidden_dim, hidden_dim)
                for i in range(num_layers)
            ])

        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Linear(lstm_out_dim, tagset_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths):
        emb = self.embedding(x)  # (batch, seq_len, embed_dim)

        out_fwd = emb
        for layer in self.layers_fwd:
            out_fwd = layer(out_fwd, lengths)

        if self.bidirectional:
            # Run backward
            rev_emb = torch.flip(emb, dims=[1])
            out_bwd = rev_emb
            for layer in self.layers_bwd:
                out_bwd = layer(out_bwd, lengths)
            out_bwd = torch.flip(out_bwd, dims=[1])
            out = torch.cat([out_fwd, out_bwd], dim=-1)
        else:
            out = out_fwd

        out = self.dropout(out)
        logits = self.fc(out)   # (batch, seq_len, tagset_size)
        return logits


def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for x_batch, y_batch, lengths in dataloader:
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        lengths = lengths.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x_batch, lengths)
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

def main():
    # 🔹 Load processed data
    (X_train, y_train), (X_val, y_val), (X_test, y_test), word2idx, tag2idx, idx2word, idx2tag = load_processed_data()

    vocab_size = len(word2idx)
    tagset_size = len(tag2idx)
    print("Vocab size:", vocab_size, "Tagset size:", tagset_size)
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Wrap in Dataset objects
    train_ds = IndexedPOSDataset(X_train, y_train)
    val_ds = IndexedPOSDataset(X_val, y_val)
    test_ds = IndexedPOSDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

    # Model, optimizer, criterion
    model = BiLSTMTagger(vocab_size=vocab_size, tagset_size=tagset_size).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=TAG_PAD)

    best_val = 0.0
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_acc = evaluate(model, val_loader, idx2tag)
        print(f"Epoch {epoch:2d} | Train loss: {train_loss:.4f} | Val acc: {val_acc:.4f}")

        # Save best checkpoint
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "model.pt"))
            print(f"  ✅ Saved best model (val acc {best_val:.4f})")

    # Final test
    print("Loading best model for test evaluation...")
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "model.pt"), map_location=DEVICE))
    test_acc = evaluate(model, test_loader, idx2tag)
    print(f"Test token-level accuracy: {test_acc:.4f}")

    # Demo inference
    demo = "The quick brown fox jumps over the lazy dog .".split()
    x_demo = [word2idx.get(w.lower(), word2idx[UNK_TOKEN]) for w in demo]
    lengths_demo = torch.tensor([len(x_demo)], dtype=torch.long)
    x_demo_tensor = torch.tensor([x_demo], dtype=torch.long).to(DEVICE)

    model.eval()
    with torch.no_grad():
        logits = model(x_demo_tensor, lengths_demo.to(DEVICE))
        preds = logits.argmax(dim=-1).squeeze(0).cpu().tolist()

    print("\nDemo tagging:")
    for tok, p in zip(demo, preds):
        # idx2tag JSON has str keys
        tag = idx2tag[str(p)] if str(p) in idx2tag else idx2tag[p]
        print(f"{tok:12} -> {tag}")


if __name__ == "__main__":
    main()
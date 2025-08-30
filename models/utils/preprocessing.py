import os
import json
import pickle
import nltk
from nltk.corpus import brown
from sklearn.model_selection import train_test_split
from collections import defaultdict

# Ensure corpus is available
nltk.download("brown")
nltk.download("universal_tagset")

# Paths
PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)


def load_brown(tagset="universal"):
    """Load POS-tagged Brown corpus."""
    return brown.tagged_sents(tagset=tagset)


def build_vocab(sentences):
    """Build vocabularies for words and tags."""
    words, tags = set(), set()

    for sent in sentences:
        for word, tag in sent:
            words.add(word.lower())
            tags.add(tag)

    # word indices start at 2 (0=PAD, 1=UNK)
    word2idx = {word: idx + 2 for idx, word in enumerate(sorted(words))}
    word2idx["<PAD>"] = 0
    word2idx["<UNK>"] = 1

    tag2idx = {tag: idx for idx, tag in enumerate(sorted(tags))}
    idx2word = {idx: word for word, idx in word2idx.items()}
    idx2tag = {idx: tag for tag, idx in tag2idx.items()}

    return word2idx, tag2idx, idx2word, idx2tag


def encode(sentences, word2idx, tag2idx, max_len=50):
    """Convert words/tags into index sequences with padding."""
    X, y = [], []
    for sent in sentences:
        words = [word.lower() for word, tag in sent]
        tags = [tag for _, tag in sent]

        word_ids = [word2idx.get(w, word2idx["<UNK>"]) for w in words]
        tag_ids = [tag2idx[t] for t in tags]

        # Pad
        if len(word_ids) < max_len:
            pad_len = max_len - len(word_ids)
            word_ids += [word2idx["<PAD>"]] * pad_len
            tag_ids += [0] * pad_len
        else:
            word_ids = word_ids[:max_len]
            tag_ids = tag_ids[:max_len]

        X.append(word_ids)
        y.append(tag_ids)

    return X, y


def preprocess_and_save(test_size=0.2, val_size=0.1, max_len=50):
    """Preprocess Brown corpus and save processed data to disk."""
    print("📥 Loading Brown corpus...")
    sentences = load_brown()

    print("🔤 Building vocabulary...")
    word2idx, tag2idx, idx2word, idx2tag = build_vocab(sentences)

    # Split dataset
    train_val, test = train_test_split(sentences, test_size=test_size, random_state=42)
    train, val = train_test_split(train_val, test_size=val_size, random_state=42)

    print("🔢 Encoding sentences...")
    X_train, y_train = encode(train, word2idx, tag2idx, max_len)
    X_val, y_val = encode(val, word2idx, tag2idx, max_len)
    X_test, y_test = encode(test, word2idx, tag2idx, max_len)

    # Save datasets
    print("💾 Saving processed data...")
    with open(os.path.join(PROCESSED_DIR, "train.pkl"), "wb") as f:
        pickle.dump((X_train, y_train), f)

    with open(os.path.join(PROCESSED_DIR, "val.pkl"), "wb") as f:
        pickle.dump((X_val, y_val), f)

    with open(os.path.join(PROCESSED_DIR, "test.pkl"), "wb") as f:
        pickle.dump((X_test, y_test), f)

    # Save vocabularies
    with open(os.path.join(PROCESSED_DIR, "word2idx.json"), "w") as f:
        json.dump(word2idx, f)

    with open(os.path.join(PROCESSED_DIR, "tag2idx.json"), "w") as f:
        json.dump(tag2idx, f)

    with open(os.path.join(PROCESSED_DIR, "idx2word.json"), "w") as f:
        json.dump(idx2word, f)

    with open(os.path.join(PROCESSED_DIR, "idx2tag.json"), "w") as f:
        json.dump(idx2tag, f)

    print("✅ Preprocessing complete! Data saved in:", PROCESSED_DIR)


if __name__ == "__main__":
    preprocess_and_save()

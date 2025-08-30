import os
import json
from collections import defaultdict, Counter
from typing import List
from models.utils.dataLoder import load_processed_data

MODEL_FILE = "saved_hmm_pos"
BASE_DIR = "/home/shivam/cs772/Assignment1"
MODEL_DIR = os.path.join(BASE_DIR,"models",MODEL_FILE)
os.makedirs(MODEL_DIR, exist_ok=True)
PROCESSED_DIR = os.path.join(BASE_DIR,"data", "processed")

class HMMPOSTagger:
    def __init__(self):
        self.tags = set()
        self.vocab = set()
        self.tag_counts = Counter()
        self.word_counts = Counter()
        self.transition_probs = defaultdict(lambda: defaultdict(float))
        self.emission_probs = defaultdict(lambda: defaultdict(float))
        self.start_probs = defaultdict(float)
    
    def train(self, tagged_sents: List[List[tuple]]):
        """
        Train HMM parameters: start_probs, transition_probs, emission_probs
        """
        print("Training HMM POS tagger...")
        total_sents = len(tagged_sents)
        for sent in tagged_sents:
            prev_tag = None
            for i, (word, tag) in enumerate(sent):
                word = word.lower()
                self.tags.add(tag)
                self.vocab.add(word)
                self.tag_counts[tag] += 1
                self.word_counts[word] += 1
                # emission counts
                self.emission_probs[tag][word] += 1
                # transition counts
                if prev_tag is None:
                    self.start_probs[tag] += 1
                else:
                    self.transition_probs[prev_tag][tag] += 1
                prev_tag = tag
        # normalize probabilities
        self.start_probs = {t: c / total_sents for t, c in self.start_probs.items()}
        for prev_tag in self.transition_probs:
            total = sum(self.transition_probs[prev_tag].values())
            self.transition_probs[prev_tag] = {t: c / total for t, c in self.transition_probs[prev_tag].items()}
        for tag in self.emission_probs:
            total = sum(self.emission_probs[tag].values())
            self.emission_probs[tag] = {w: c / total for w, c in self.emission_probs[tag].items()}
        print("Training complete.")

    def predict(self, words: List[str]) -> List[str]:
        words = [w.lower() for w in words]
        T = len(words)
        tags = list(self.tags)
        N = len(tags)
        
        viterbi = [{} for _ in range(T)]
        backpointer = [{} for _ in range(T)]
        
        # initialization
        for tag in tags:
            emission = self.emission_probs[tag].get(words[0], 1e-6)
            viterbi[0][tag] = self.start_probs.get(tag, 1e-6) * emission
            backpointer[0][tag] = None
        
        # recursion
        for t in range(1, T):
            for tag in tags:
                max_prob = 0
                best_prev = None
                emission = self.emission_probs[tag].get(words[t], 1e-6)
                for prev_tag in tags:
                    trans_prob = self.transition_probs[prev_tag].get(tag, 1e-6)
                    prob = viterbi[t-1][prev_tag] * trans_prob * emission
                    if prob > max_prob:
                        max_prob = prob
                        best_prev = prev_tag
                viterbi[t][tag] = max_prob
                backpointer[t][tag] = best_prev

        best_path = []
        last_tag = max(viterbi[T-1], key=viterbi[T-1].get)
        best_path.append(last_tag)
        for t in range(T-1, 0, -1):
            last_tag = backpointer[t][last_tag]
            best_path.append(last_tag)
        best_path.reverse()
        return best_path

    def save(self, path=MODEL_DIR):
        data = {
            "tags": list(self.tags),
            "vocab": list(self.vocab),
            "start_probs": self.start_probs,
            "transition_probs": {k: dict(v) for k, v in self.transition_probs.items()},
            "emission_probs": {k: dict(v) for k, v in self.emission_probs.items()},
        }
        with open(os.path.join(path, "hmm_pos.json"), "w") as f:
            json.dump(data, f)
        print(f"Model saved to {path}/hmm_pos.json")
    
    @classmethod
    def load(cls, path=MODEL_DIR):
        with open(os.path.join(path, "hmm_pos.json")) as f:
            data = json.load(f)
        model = cls()
        model.tags = set(data["tags"])
        model.vocab = set(data["vocab"])
        model.start_probs = data["start_probs"]
        model.transition_probs = defaultdict(lambda: defaultdict(float), 
                                             {k: defaultdict(float, v) for k, v in data["transition_probs"].items()})
        model.emission_probs = defaultdict(lambda: defaultdict(float),
                                           {k: defaultdict(float, v) for k, v in data["emission_probs"].items()})
        return model

if __name__ == "__main__":

    (X_train, y_train), (X_val, y_val), (X_test, y_test), word2idx, tag2idx, idx2word, idx2tag = load_processed_data()

    train_sents = [
        [(idx2word[str(w)], idx2tag[str(t)]) for w, t in zip(words, tags) if w != 0]  # skip PAD
        for words, tags in zip(X_train, y_train)
    ]
    test_sents = [
        [(idx2word[str(w)], idx2tag[str(t)]) for w, t in zip(words, tags) if w != 0]
        for words, tags in zip(X_test, y_test)
    ]

    hmm = HMMPOSTagger()
    hmm.train(train_sents)
    hmm.save()

    # Load saved model
    hmm2 = HMMPOSTagger.load()

    sentence = "The quick brown fox jumps over the lazy dog .".split()
    tags = hmm2.predict(sentence)
    for w, t in zip(sentence, tags):
        print(f"{w:12} -> {t}")
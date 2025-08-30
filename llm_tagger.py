import os
import nltk
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

nltk.download("punkt", quiet=True)

class POSTagger:
    def __init__(self, model_name: str = "vblagoje/bert-english-uncased-finetuned-pos",
                 local_dir: str = "saved_pos_model"):
        """
        Initializes the POS Tagger.
        - Loads model from local_dir if exists
        - Otherwise downloads from Hugging Face and saves locally
        """
        self.model_name = model_name
        self.local_dir = local_dir
        os.makedirs(self.local_dir, exist_ok=True)

        if os.listdir(self.local_dir):
            print("Loading model from local directory...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.local_dir)
            self.model = AutoModelForTokenClassification.from_pretrained(self.local_dir)
        else:
            print("Downloading model from Hugging Face...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForTokenClassification.from_pretrained(self.model_name)
            self.tokenizer.save_pretrained(self.local_dir)
            self.model.save_pretrained(self.local_dir)
            print(f"Model saved locally at '{self.local_dir}'")

        # Create pipeline
        self.pipeline = pipeline(
            "token-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy="simple"
        )

    def tag(self, text: str):
        """
        Performs POS tagging on a given sentence.
        Returns a list of tuples: (word, POS_tag)
        """
        results = self.pipeline(text)
        return [(item['word'], item['entity_group']) for item in results]

if __name__ == "__main__":
    tagger = POSTagger()
    sentence = "The quick brown fox jumps over the lazy dog."
    tags = tagger.tag(sentence)

    print("\nPOS tagging results:\n")
    for word, tag in tags:
        print(f"{word:<15} → {tag}")

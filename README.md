# POS Tagging Project

## 📌 Overview
This project focuses on **Part-of-Speech (POS) tagging**, the task of assigning a grammatical category (noun, verb, adjective, etc.) to each word in a sentence.  
The assignment involves building, evaluating, and comparing three models:  

1. **Hidden Markov Model (HMM)**  
2. **RNN-based Encoder–Decoder Model** (RNN/GRU/LSTM)  
3. **Large Language Model (LLM)**  

---

## 🎯 Objectives
- Implement three different approaches for POS tagging.  
- Evaluate models on a test set using **accuracy, precision, recall, and F1-score**.  
- Generate and analyze a **confusion matrix** for POS tags.  
- Compare models, identifying their **strengths and weaknesses**.  

---

## 📂 Project Structure

```bash
    pos_tagging_project/
    │
    ├── data/ # Dataset storage
    │ ├── raw/ # Original dataset (train/test)
    │ ├── processed/ # Preprocessed tokenized data
    │ └── sample_sentences.txt # Example input sentences
    │
    ├── models/ # Model implementations
    │ ├── hmm.py # Hidden Markov Model
    │ ├── rnn.py # RNN/GRU/LSTM encoder-decoder
    │ └── llm.py # LLM-based tagging
    │
    ├── evaluation/ # Evaluation scripts
    │ ├── metrics.py # Accuracy, Precision, Recall, F1
    │ ├── confusion_matrix.py # Confusion matrix plotting
    │ └── compare_models.py # Side-by-side comparison
    │
    ├── utils/ # Helper functions
    │ ├── preprocessing.py # Tokenization, vocab, train-test split
    │ ├── data_loader.py # Dataset loading
    │ └── visualization.py # Plotting utilities
    │
    ├── notebooks/ # Jupyter notebooks for experiments
    │
    ├── results/ # Results storage
    │ ├── logs/ # Training logs
    │ ├── metrics/ # Metrics (json/csv)
    │ └── plots/ # Confusion matrices, graphs
    │
    ├── main.py # Entry point
    ├── requirements.txt # Dependencies
    ├── README.md # Documentation
    └── report.md # Final analysis write-up
```


## ⚙️ Setup

1. Clone this repository:
```bash
git clone https://github.com/your-username/pos-tagging-project.git
cd pos-tagging-project
```
2. Create a virtual environment and install dependencies
```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```
3. Add your dataset in the data/raw/ folder and preprocess it
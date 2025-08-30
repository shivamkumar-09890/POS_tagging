import os
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request

import torch
from models.rnn import BiLSTMTagger, collate_batch, load_vocabs, TAG_PAD, DEVICE
from models.hmm import HMMPOSTagger
from llm_tagger import POSTagger  # your LLM wrapper

# -------------------- Setup -------------------- #
BASE_DIR = "/home/shivam/cs772/Assignment1"
MODEL_DIR_RNN = os.path.join(BASE_DIR, "saved_rnn_pos/model.pt")
MODEL_DIR_HMM = os.path.join(BASE_DIR, "saved_hmm_pos")
MODEL_DIR_LLM = os.path.join(BASE_DIR, "saved_pos_model")
DATA_DIR = os.path.join(BASE_DIR, "data/processed")

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# -------------------- Load Models -------------------- #
# Load vocabs for RNN
word2idx, tag2idx, idx2word, idx2tag = load_vocabs(DATA_DIR)
vocab_size = len(word2idx)
tagset_size = len(tag2idx)

# Load RNN
rnn_model = BiLSTMTagger(vocab_size=vocab_size, tagset_size=tagset_size).to(DEVICE)
rnn_model.load_state_dict(torch.load(MODEL_DIR_RNN, map_location=DEVICE))
rnn_model.eval()

# Load HMM
hmm_model = HMMPOSTagger.load(MODEL_DIR_HMM)

# Load LLM
llm_model = POSTagger(local_dir=MODEL_DIR_LLM)

# -------------------- Helper Functions -------------------- #
def tag_rnn(sentence: str):
    words = sentence.split()
    X = [[word2idx.get(w.lower(), 1) for w in words]]  # 1=UNK
    lengths = torch.tensor([len(words)]).to(DEVICE)
    x_batch = torch.tensor(X).to(DEVICE)
    with torch.no_grad():
        logits = rnn_model(x_batch, lengths)
        preds = logits.argmax(dim=-1)[0].tolist()
    return [(w, idx2tag[str(p)]) for w, p in zip(words, preds)]

def tag_hmm(sentence: str):
    words = sentence.split()
    preds = hmm_model.predict(words)
    return list(zip(words, preds))

# -------------------- Routes -------------------- #
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/tag", response_class=JSONResponse)
async def tag_sentence(sentence: str = Form(...)):
    rnn_tags = tag_rnn(sentence)
    hmm_tags = tag_hmm(sentence)
    llm_tags = llm_model.tag(sentence)

    return {
        "sentence": sentence,
        "rnn": rnn_tags,
        "hmm": hmm_tags,
        "llm": llm_tags
    }

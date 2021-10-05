import streamlit as st

import torch
from torch._C import device
import torch.nn as nn

from os.path import exists
import gdown

from notebooks.utils.processing import get_index_toxic_words, color_toxic_words, f1
from notebooks.utils.lstm import spacy_tokenizer, get_vocab

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

"""
# SemEval2021
"""

text = """
<div style="text-align: justify"> 
Pendiente
"""

st.markdown(text, unsafe_allow_html=True)

class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, stacked_layers, dropout_p, weight, hidden_dim, vocab_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim         
        self.stacked_layers = stacked_layers 
        
        self.word_embeddings = nn.Embedding.from_pretrained(weight)
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=stacked_layers,
                            dropout=dropout_p,
                            bidirectional=True)
        # Linear layers
        self.fc1 = nn.Linear(hidden_dim*2, 1) 

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        output, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        x = torch.sigmoid(self.fc1(output.view(len(sentence), -1)))
        return x

def prepare_sequence(vocab, seq):
    idxs = vocab.lookup_indices(seq)      # Si no está lo pone como 0
    return torch.tensor(idxs, dtype=torch.long, device=device)

def prepare_sequence_tags(seq):
    tag_to_ix = {"non_toxic": 0, "toxic": 1} 
    idxs = [tag_to_ix[s] for s in seq]
    return torch.tensor(idxs, dtype=torch.long, device=device)

def tagger_LSTM(model, text, threshold=0.5):
    """
    Performs the tagging with the LSTM model we trained.
    """
    # ix_to_tag = {0: 'non_toxic', 1: 'toxic'}
    words = spacy_tokenizer(text.lower())
    
    with torch.no_grad():
        inputs = prepare_sequence(words)
        tag_scores = model(inputs)
        
        tags = [1 if x > threshold else 0 for x in tag_scores]
        tagged_sentence = list(zip(words, tags))

    return tagged_sentence

@st.cache(allow_output_mutation=True)
def load_model():
    model_path = 'notebooks/models/best-model.pt'

    if not exists(model_path):
        # Download the model
        url = 'https://drive.google.com/uc?id=1KO-QXUBfwzjauWLhiVi9StD3y0GtiBbj'
        gdown.download(url, model_path, quiet=False)

    model = torch.load(model_path)
    model.to(device)

    return model
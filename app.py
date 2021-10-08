import streamlit as st

import torch
import torch.nn as nn

import pandas as pd
from os.path import exists
import gdown
import branca.colormap as cm

from notebooks.utils.lstm import spacy_tokenizer, get_vocab

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

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

def tagger_LSTM(model, vocab, text):
    """
    Performs the tagging with the LSTM model we trained.
    """
    # ix_to_tag = {0: 'non_toxic', 1: 'toxic'}
    # words = spacy_tokenizer(text.lower())
    words = text.split()
    if text:
        with torch.no_grad():
            inputs = prepare_sequence(vocab, words)
            tag_scores = model(inputs).cpu().numpy().reshape(-1)
            tagged_sentence = list(zip(words, tag_scores))

        return tagged_sentence
    else:
        return []

def highlight_toxic_words(tagged_sentence):
    colormap = cm.LinearColormap(colors=['#FFFFFF', '#FF0000'], vmin=0.0, vmax=1.0)
    colored_string = ''
    for (word, score) in tagged_sentence:
        colored_string += f'<span style="background-color: {colormap(score)}">{word}</span> '

    return colored_string

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

# Cache the function to load the embeddings
get_vocab = st.cache(get_vocab, allow_output_mutation=True)

"""
# SemEval 2021 Task 5: Toxic Spans Detection
"""
st.sidebar.image('images/toxic.jpg')
description = """
*Model to detect toxic spans for the [SemEval 2021](https://competitions.codalab.org/competitions/25623) competition.*
"""

st.sidebar.markdown(description)

text = """
<div style="text-align: justify"> 
Moderation is crucial to promoting healthy online discussions. Although several toxicity (abusive language) detection datasets and 
models have been released, most of them classify whole comments or documents, and do not identify the spans that make a text toxic. 
But highlighting such toxic spans can assist human moderators (e.g., news portals moderators) who often deal with lengthy comments, 
and who prefer attribution instead of just a system-generated unexplained toxicity score per post. The evaluation of systems that could 
accurately locate toxic spans within a text is thus a crucial step towards successful semi-automated moderation.
</div>
"""

st.markdown(text, unsafe_allow_html=True)

"""
## Shared task
"""

text = """
<div style="text-align: justify"> 
As a complete submission for the Shared Task, systems will have to extract a list of toxic spans, or an empty list, per text. As toxic 
span we define a sequence of words that attribute to the text's toxicity.
<br><br>
First we tried with a HMM and a CRF, but both of them had poor results. Then we tried with a LSTM to make the tagging and it worked
better. You can try the model here (the more red a word is highlighted the more toxic it is):
<br><br>
</div>
"""

st.markdown(text, unsafe_allow_html=True)

train_df = pd.read_csv('data/tsd_train.csv')
vocab = get_vocab(train_df)
model = load_model()

input_text = st.text_input(label='Text', value='This is a stupid and kinda silly example, try your own toxic text here', max_chars=150)

tagged_sentence = tagger_LSTM(model, vocab, input_text)
highlighted_sentence = highlight_toxic_words(tagged_sentence)
st.markdown(highlighted_sentence, unsafe_allow_html=True)

text = """
<div style="text-align: justify"> 
<br><br>
The F1 score of the LSTM in the evaluation dataset was 64.88, which was rather low. However, even the winners of this task just managed to get around 70. 
We believe the low scores are due to the fact the training dataset was poorly tagged.
"""

st.markdown(text, unsafe_allow_html=True)
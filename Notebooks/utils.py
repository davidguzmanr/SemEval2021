import string
from termcolor import colored
from itertools import chain
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import sklearn
import pycrfsuite
from tqdm import tqdm

import spacy
from spacy.symbols import ORTH, LEMMA, POS
nlp = spacy.load("en_core_web_md")

# SpaCy hace cosas no deseadas con algunas palabras al tokenizar, como don't -> [do, n't], pero se puede corregir.
# Pero de acuerdo a SpaCy esa es la convención, además, eso se debería codificar en los embeddings, así que se quede
# así, sólo hay que usar el mismo tokenizador en Field de torchtext (permite el de SpaCy entre otros).
# nlp.tokenizer.add_special_case("don't", [{ORTH: "do"}, {ORTH: "not"}])
# nlp.tokenizer.add_special_case("don't", [{ORTH: "don't"}])
# nlp.tokenizer.add_special_case("doesn't", [{ORTH: "does"}, {ORTH: "not"}])

def spacy_tokenizer	(text):
    return [str(token) for token in nlp(text)]

def color_toxic_words(index, text):
    colored_string = ''
    for i,x in enumerate(text):
        if i in index:
            colored_string += colored(x, on_color='on_red')
        else:
            colored_string += colored(x)
            
    return colored_string

def remove_symbols(index, text):
    """
    Remueve los índices que corresponden a símbolos 'no tóxicos', como espacios en blanco
    comas, puntos, etc.
    """
    index_clean = []   
    for i in index:
        x = text[i]
        if x not in ('"\()+,-./:;<=>[\\]^_`{|}~' + string.whitespace):
            index_clean.append(i)
                        
    return index_clean

def completely_toxic(span, text):
    if span == []:
        return [i for i in range(len(text))]
    else:
    	return span

def separate_words(indices):
    """
    Separa los índices por palabras.
    """

    toxic_words_indices = []
    m = 0
    for i,(j,k) in enumerate(zip(indices[0:-1], indices[1:])):
        if k-j != 1:
            toxic_words_indices.append(indices[m:i+1])
            m = i+1
    toxic_words_indices.append(indices[m:]) # Última palabra
    
    return toxic_words_indices   


def postprocessing(indices_list, delta=7):
    """
    Pone como tóxicos los caracteres en medio de dos palabras tóxicas si el espacio
    entre ellas es menor a delta.
    """

    # Asumiendo que tienes indices numéricos enteros.
    if len(indices_list) > 1:
        l = sorted(indices_list)
        new_list = []
        for i in range(len(indices_list)-1):
            # Agrego el indice existente
            new_list.append(l[i])
            # Si no hay mucho espacio entre este y el siguiente indice, selecciono todos los indices intermedios
            if (l[i+1] - l[i]) <= delta:
                new_list = new_list + list(range(l[i]+1,l[i+1]))
                
        new_list.append(l[-1]) # El ultimo elemento
        return new_list
    else:
        return indices_list


def get_index_toxic_words(sentence, tagged_sentence, delta=7):
    toxic_indices = []   
    m = 0
    #tag_to_ix = {"non_toxic": 0, "toxic": 1}
    for word_tag in tagged_sentence:
        word, tag = word_tag    
        if tag == 1: #toxic
            # Si la palabra tóxica aparece 2 o más veces ésto solo dará la primera 
            # aparición, hay que arreglar eso pero por lo mientras sirve
            # word_indices = [sentence.find(word) + i for i in range(len(word))]
            # toxic_indices.append(word_indices)
            
            # Así parece evitar el problema de la palabra repetida
            word_indices = [m + sentence[m:].find(word) + i for i in range(len(word))]
            toxic_indices.append(word_indices)
        # Ya se arregla el 'bug' de 'stupidity'
        m += sentence[m:].find(word) + len(word)
            
    toxic_indices = [val for sublist in toxic_indices for val in sublist]
    
    # Unir espacios y otras cosas para que suba el F1    
    return postprocessing(toxic_indices, delta)


def f1(predictions, gold):
    """
    F1 (a.k.a. DICE) operating on two lists of offsets (e.g., character).
    >>> assert f1([0, 1, 4, 5], [0, 1, 6]) == 0.5714285714285714
    :param predictions: a list of predicted offsets
    :param gold: a list of offsets serving as the ground truth
    :return: a score between 0 and 1
    """
    if len(gold) == 0:
        return 1. if len(predictions) == 0 else 0.
    if len(predictions) == 0:
        return 0.
    predictions_set = set(predictions)
    gold_set = set(gold)
    nom = 2 * len(predictions_set.intersection(gold_set))
    denom = len(predictions_set) + len(gold_set)
    return float(nom)/float(denom)

def f1_scores(pred, true_index, tokenized, text, threshold=0.5):
    scores_LSTM = 0
    for i in range(len(pred)):
        tags = [1 if x > threshold else 0 for x in pred[i]]
        tagged_sentence = list(zip(tokenized[i], tags))
        prediction_index = get_index_toxic_words(text[i], tagged_sentence)
        scores_LSTM += f1(prediction_index, true_index[i])
    return scores_LSTM/len(pred)


def plot_loss_and_score(train_loss, test_loss, f1_scores_train, f1_scores_test, show=True):
    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(18,7))

    ax0.plot(np.arange(1, len(train_loss) + 1), train_loss, marker='o', label='Train loss')
    ax0.plot(np.arange(1, len(test_loss) + 1), test_loss, marker='o', label='Test loss')
    ax0.set_xlabel(r'\textbf{Epochs}',size=16)
    ax0.set_ylabel(r'\textbf{Loss}', size=16)
    ax0.tick_params(labelsize=14)
    ax0.legend(fontsize=14)

    ax1.plot(np.arange(1, len(f1_scores_train) + 1), f1_scores_train, 
             marker='o', label='F1 score in train')
    ax1.plot(np.arange(1, len(f1_scores_test) + 1), f1_scores_test, 
             marker='o', label='F1 score in test')
    ax1.set_xlabel(r'\textbf{Epochs}',size=16)
    ax1.set_ylabel(r'\textbf{F1 score}', size=16)
    ax1.tick_params(labelsize=14)
    ax1.legend(fontsize=14)
    
    title = 'train-F1: {:.4f} \n test-F1: {:.4f}'.format(np.max(f1_scores_train), np.max(f1_scores_test))
    ax1.set_title(title, fontweight='bold', size=16)


    if show:
        plt.show()



##################################################
# Lo siguiente es para lo de CRF, que ya no usamos

def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    features = [
        'bias',
        'word.lower=' + word.lower(),
        'word[-3:]=' + word[-3:],
        'word[-2:]=' + word[-2:],
        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'word.isdigit=%s' % word.isdigit(),
        'postag=' + postag,
        'postag[:2]=' + postag[:2],
    ]
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.extend([
            '-1:word.lower=' + word1.lower(),
            '-1:word.istitle=%s' % word1.istitle(),
            '-1:word.isupper=%s' % word1.isupper(),
            '-1:postag=' + postag1,
            '-1:postag[:2]=' + postag1[:2],
        ])
    else:
        features.append('BOS')
        
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.extend([
            '+1:word.lower=' + word1.lower(),
            '+1:word.istitle=%s' % word1.istitle(),
            '+1:word.isupper=%s' % word1.isupper(),
            '+1:postag=' + postag1,
            '+1:postag[:2]=' + postag1[:2],
        ])
    else:
        features.append('EOS')
                
    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]

def token_postag_label(sentence):
    return pos_tag(word_tokenize(sentence))

def f1(predictions, gold):
    """
    F1 (a.k.a. DICE) operating on two lists of offsets (e.g., character).
    >>> assert f1([0, 1, 4, 5], [0, 1, 6]) == 0.5714285714285714
    :param predictions: a list of predicted offsets
    :param gold: a list of offsets serving as the ground truth
    :return: a score between 0 and 1
    """
    if len(gold) == 0:
        return 1. if len(predictions) == 0 else 0.
    if len(predictions) == 0:
        return 0.
    predictions_set = set(predictions)
    gold_set = set(gold)
    nom = 2 * len(predictions_set.intersection(gold_set))
    denom = len(predictions_set) + len(gold_set)
    return float(nom)/float(denom)
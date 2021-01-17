import string
from termcolor import colored
from itertools import chain
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import sklearn
import pycrfsuite
from tqdm import tqdm

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
        if x not in (string.punctuation + string.whitespace):
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
    toxic_words_indices.append(indices[m:])
    
    return toxic_words_indices   

def get_index_toxic_words(sentence, tagged_sentence):
    toxic_indices = []   
    m = 0
    for word_tag in tagged_sentence:
        word, tag = word_tag    
        if tag == 'toxic':
            # Si la palabra tóxica aparece 2 o más veces ésto solo dará la primera 
            # aparición, hay que arreglar eso pero por lo mientras sirve
            # word_indices = [sentence.find(word) + i for i in range(len(word))]
            # toxic_indices.append(word_indices)
            
            # Así parece evitar el problema de la palabra repetida
            word_indices = [m + sentence[m:].find(word) + i for i in range(len(word))]
            toxic_indices.append(word_indices)
            m += sentence[m:].find(word) + len(word) + 1
            
    toxic_indices = [val for sublist in toxic_indices for val in sublist]
        
    return toxic_indices

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
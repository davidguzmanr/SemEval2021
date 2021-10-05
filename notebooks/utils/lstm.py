import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.data import Field

from .processing import separate_words, f1_scores

import spacy
import ast

from tqdm import tqdm
from IPython.display import clear_output

nlp = spacy.load('en_core_web_md')
dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True

# SpaCy hace cosas no deseadas con algunas palabras al tokenizar, como don't -> [do, n't], pero se puede corregir.
# Pero de acuerdo a SpaCy esa es la convención, además, eso se debería codificar en los embeddings, así que se quede
# así, sólo hay que usar el mismo tokenizador en Field de torchtext (permite el de SpaCy entre otros).

# from spacy.symbols import ORTH, LEMMA, POS
# nlp.tokenizer.add_special_case("don't", [{ORTH: "do"}, {ORTH: "not"}])
# nlp.tokenizer.add_special_case("don't", [{ORTH: "don't"}])
# nlp.tokenizer.add_special_case("doesn't", [{ORTH: "does"}, {ORTH: "not"}])

def spacy_tokenizer	(text):
    return [str(token) for token in nlp(text)]

def prepare_data(spans, texts):
    data = []
    for index, text in tqdm(zip(spans, texts), total=len(texts)):
        toxic_words = [text[i[0]:i[-1]+1] for i in separate_words(index) if len(index) > 0]
        
        tokens = spacy_tokenizer(text)
        tagged_tokens = []
        
        for token in tokens:
            if token in toxic_words:
                tagged_tokens.append('toxic')
                # Removemos en caso de que se repita posteriormente pero esté como 'non_toxic'
                toxic_words.remove(token) 
            else:
                tagged_tokens.append('non_toxic')
                
        data.append((tokens, tagged_tokens, text, index))

    return data

def get_vocab(train_df):
    train_df['text'] = train_df['text'].apply(lambda x:x.lower())

    # Aquí había un problema, estábamos usando 2 tokenizadores diferentes para sacar los
    # embeddings y para preprocesar el texto para entrenar. Pondré el de SpaCy como 
    # tokenizador en común con el corpus de 'en_core_web_md'
    text_field = Field(
        tokenize='spacy',
        tokenizer_language='en_core_web_md',
        lower=True
    )
    # sadly have to apply preprocess manually
    preprocessed_text = train_df['text'].apply(lambda x: text_field.preprocess(x))
    # load fastext simple embedding with 200d
    text_field.build_vocab(
        preprocessed_text, 
        vectors='glove.twitter.27B.200d'
    )
    # get the vocab instance
    vocab = text_field.vocab

    return vocab

def plot_loss_and_score(train_loss, test_loss, f1_scores_train, f1_scores_test, show=True):
    _, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(18,7))

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

# WTF Mario, this is a mess
def train_model(model, trainloader, testloader, stop_after_best, savefile):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())

    loss_per_epoch = [0]
    training_loss = [0]
    f1_scores_train = [0]
    f1_scores_dev = [0]
    best_l = None
    best_tl = None
    worst_l = None
    worst_tl = None
    worst_l_f1 = None
    best_l_f1 = None
    worst_tl_f1 = None
    last_epoch_save = 0

    epochs_without_change = 0
    epochs = len(loss_per_epoch)

    while epochs_without_change < stop_after_best:
        clear_output(wait=True)

        print("Training on: " + torch.cuda.get_device_name(torch.cuda.current_device()))
        print("###############################################")
        print("Current epoch: " + str(epochs))
        print("Last model save was in epoch " + str(last_epoch_save))
        print("Stopping training in: " + str(stop_after_best - epochs_without_change) + " epochs.")
        print("###############################################")
        print("[Best iter] training F1 is: " + str(best_tl))
        print("[Best iter] dev F1 is: " + str(best_l))
        print("###############################################")
        print("[Last iter] training F1 was: " + str(f1_scores_train[-1]))
        print("[Last iter] dev. F1 was: " + str(f1_scores_dev[-1]))
        print("###############################################")
        
        # Dibujo lo que puedo
        plot_loss_and_score(training_loss, loss_per_epoch, f1_scores_train, f1_scores_dev, show=True)
        
        tl = 0
        t_pred_l = []
        t_true_index_l = []
        t_tokenized_l = []
        t_text_l = []

        for _, v in tqdm(enumerate(trainloader), total=len(trainloader)): # Not using batches yet
            text = torch.reshape(v['text'], (-1,))
            tags = torch.reshape(v['spans'], (-1,))
            optimizer.zero_grad()
            tag_scores = model(text)
            
            # Para la F1
            t_pred_l.append(tag_scores.cpu().detach().numpy())
            t_true_index_l.append([a.cpu().detach().numpy()[0] for a in v['true_index']])
            t_tokenized_l.append([a[0] for a in v['tokenized']])
            t_text_l.append(v['original_text'][0])
            
            loss = criterion(torch.reshape(tag_scores, (-1,)), torch.reshape(tags, (-1,)).float())
            tl += loss.item()
            loss.backward()
            optimizer.step()

        tl /= len(trainloader)
        l = 0
        print("Starting evaluation for loss function.")
        # evaluar el modelo
        pred_l = []
        true_index_l = []
        tokenized_l = []
        text_l = []
        
        model.eval()
        with torch.no_grad():
            for v in testloader:
                text = torch.reshape(v['text'], (-1,))
                tags = torch.reshape(v['spans'], (-1,))

                tag_scores = model(text)
                
                #Para la F1
                pred_l.append(tag_scores.cpu().detach().numpy())
                true_index_l.append([a.cpu().detach().numpy()[0] for a in v['true_index']])
                tokenized_l.append([a[0] for a in v['tokenized']])
                text_l.append(v['original_text'][0])
                
                loss = criterion(torch.reshape(tag_scores, (-1,)), torch.reshape(tags, (-1,)).float())
                l += loss.item()
        
        model.train()
        l /= len(testloader)
        print("Starting evaluation for dev F1")
        f1_d = f1_scores(pred_l, true_index_l, tokenized_l, text_l)
        # Es aproximado, pero solo es una referencia
        f1_t = f1_scores(t_pred_l, t_true_index_l, t_tokenized_l, t_text_l) 
        
        
        epochs_without_change += 1
        if best_l is None or best_l < f1_d:
            print("Model improved, saving.")
            torch.save(model, savefile)
            best_l = f1_d
            best_tl = f1_t
            epochs_without_change = 0
            last_epoch_save = epochs
            print("Model improved, saved.")

        # Para graficar con una escala coherente.
        if(worst_l_f1 is None or f1_d < worst_l_f1):
            worst_l_f1 = f1_d
            f1_scores_dev[0] = worst_l_f1
        if(worst_tl_f1 is None or f1_t < worst_tl_f1):
            worst_tl_f1 = f1_t
            f1_scores_train[0] = worst_tl_f1
        if(worst_tl is None or tl > worst_tl):
            worst_tl = tl
            training_loss[0] = worst_tl
        if(worst_l is None or l > worst_l):
            worst_l = l
            loss_per_epoch[0] = worst_l

        # Rastreo las perdidas
        loss_per_epoch.append(l)
        training_loss.append(tl)
        f1_scores_train.append(f1_t)
        f1_scores_dev.append(f1_d)
        # Rastreo la época actual
        epochs += 1
    print('Finished Training')

    return loss_per_epoch, training_loss, f1_scores_train, f1_scores_dev
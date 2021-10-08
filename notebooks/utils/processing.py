from termcolor import colored
import string

def color_toxic_words(index, text, html=False):
    if not html:
        colored_string = ''
        for i, x in enumerate(text):
            if i in index:
                colored_string += colored(x, on_color='on_red')
            else:
                colored_string += colored(x)
    else:
        colored_string = ''
        for i, x in enumerate(text):
            if i in index:
                colored_string += f'<span style="background-color: #FF0000">{x}</span>'
            else:
                colored_string += x
            
    return colored_string

def remove_symbols(index, text):
    """
    Remueve los índices que corresponden a símbolos 'no tóxicos', como espacios en blanco
    comas, puntos, etc.
    """
    index_clean = []   
    for i in index:
        x = text[i]
        if x not in ('"()+,-./:;<=>[\\]^_`{|}~' + string.whitespace):
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
    scores = 0
    for i in range(len(pred)):
        tags = [1 if x > threshold else 0 for x in pred[i]]
        tagged_sentence = list(zip(tokenized[i], tags))
        prediction_index = get_index_toxic_words(text[i], tagged_sentence)
        scores += f1(prediction_index, true_index[i])
    return scores/len(pred)
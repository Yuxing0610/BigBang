import numpy as np
from nltk.tokenize import TweetTokenizer, word_tokenize


def get_character_label(character):
    '''
    FUNCTION:
    get the label of speaker for training, all characters except for the main characters are merged in to the "Secondary" Label
    '''
    main_characters = ["Sheldon", "Penny", "Leonard", "Raj", "Howard", "Amy", "Bernadette", "End"]
    if character in main_characters:
        return character
    else:
        return "Secondary"
    

def aggregate_vector_list(vlist, aggfunc):
    '''
    FUNCTION:
    aggregate a set of word vectors to a single vector represents a sentence
    '''
    if aggfunc == 'max':
        return np.array(vlist).max(axis=0)
    elif aggfunc == 'min':
        return np.array(vlist).min(axis=0)
    elif aggfunc == 'mean':
        return np.array(vlist).mean(axis=0)
    else:
        return np.zeros(np.array(vlist).shape[1])
    

def word2text(documents, vector_dict, dim, method):
    '''
    FUNCTION:
    utilize the aggregate_vector_list() function to get the representations of all sentences in a document.
    '''
    tknzr = TweetTokenizer()
    aggregated_doc_vectors = np.zeros((len(documents), dim))
    for index, doc in enumerate(documents):
        vlist = [vector_dict[token] for token in tknzr.tokenize(doc) if token in vector_dict]
        if(len(vlist) < 1):
            continue 
        else:
            aggregated_doc_vectors[index] = aggregate_vector_list(vlist, method)
    return aggregated_doc_vectors
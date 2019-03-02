import nltk
import pickle
import re
import numpy as np

nltk.download('stopwords')
from nltk.corpus import stopwords

# Paths for all resources for the bot.
RESOURCE_PATH = {
    'INTENT_RECOGNIZER': 'intent_recognizer.pkl',
    'TAG_CLASSIFIER': 'tag_classifier.pkl',
    'TFIDF_VECTORIZER': 'tfidf_vectorizer.pkl',
    'THREAD_EMBEDDINGS_FOLDER': 'thread_embeddings_by_tags',
    'WORD_EMBEDDINGS': 'starspace_embedding.tsv',
}


def text_prepare(text):
    """Performs tokenization and simple preprocessing."""
    
    replace_by_space_re = re.compile('[/(){}\[\]\|@,;]')
    bad_symbols_re = re.compile('[^0-9a-z #+_]')
    stopwords_set = set(stopwords.words('english'))

    text = text.lower()
    text = replace_by_space_re.sub(' ', text)
    text = bad_symbols_re.sub('', text)
    text = ' '.join([x for x in text.split() if x and x not in stopwords_set])

    return text.strip()


def load_embeddings(embeddings_path):
    """Loads pre-trained word embeddings from tsv file.

    Args:
      embeddings_path - path to the embeddings file.

    Returns:
      embeddings - dict mapping words to vectors;
      embeddings_dim - dimension of the vectors.
    """
    
    # Hint: you have already implemented a similar routine in the 3rd assignment.
    # Note that here you also need to know the dimension of the loaded embeddings.
    # When you load the embeddings, use numpy.float32 type as dtype

    ########################
    #### YOUR CODE HERE ####
    ########################
    starspace_embeddings = dict()
    with open(embeddings_path, encoding='utf-8') as f:
        for line in f.readlines():
            row = line.strip().split('\t')
            starspace_embeddings[row[0]] = np.array(row[1:], dtype=np.float32)
    # Dimension of loaded embeddings
    starspace_embeddings_dim = starspace_embeddings[list(starspace_embeddings)[0]].shape[0]
    
    return starspace_embeddings, starspace_embeddings_dim


def question_to_vec(question, embeddings, dim):
    """Transforms a string to an embedding by averaging word embeddings."""
    
    # Hint: you have already implemented exactly this function in the 3rd assignment.

    ########################
    #### YOUR CODE HERE ####
    ########################
    word_tokens = question.split(" ")
    question_len = len(word_tokens)
    question_mat = np.zeros((question_len,dim), dtype = np.float32)
    
    for idx, word in enumerate(word_tokens):
        if word in embeddings:
            question_mat[idx,:] = embeddings[word]
            
    # remove zero-rows which stand for OOV words       
    question_mat = question_mat[~np.all(question_mat == 0, axis = 1)]
    
    # Compute the mean of each word along the sentence
    if question_mat.shape[0] > 0:
        vec = np.array(np.mean(question_mat, axis = 0), dtype = np.float32).reshape((1,dim))
    else:
        vec = np.zeros((1,dim), dtype = np.float32)
        
    return vec


def unpickle_file(filename):
    """Returns the result of unpickling the file content."""
    with open(filename, 'rb') as f:
        return pickle.load(f)

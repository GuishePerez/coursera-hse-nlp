#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 12:12:57 2019

@author: Guishe
"""

import sys
sys.path.append("..")
from common.download_utils import download_week1_resources

download_week1_resources()

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

from ast import literal_eval
import pandas as pd
import numpy as np

def read_data(filename):
    data = pd.read_csv(filename, sep='\t')
    data['tags'] = data['tags'].apply(literal_eval)
    return data

train = read_data('data/train.tsv')
validation = read_data('data/validation.tsv')
test = pd.read_csv('data/test.tsv', sep='\t')

train.head()


X_train, y_train = train['title'].values, train['tags'].values
X_val, y_val = validation['title'].values, validation['tags'].values
X_test = test['title'].values


import re
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def text_prepare(text):
    """
        text: a string
        
        return: modified initial string
    """
    # lowercase text
    text = text.lower()
    # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = re.sub(REPLACE_BY_SPACE_RE, ' ', text)
    # delete symbols which are in BAD_SYMBOLS_RE from text
    text = re.sub(BAD_SYMBOLS_RE, '', text)
    # delete stopwords from text
    text = " ".join([word for word in text.split() if word not in STOPWORDS])
#    text = text.strip()
    return text

def test_text_prepare():
    examples = ["SQL Server - any equivalent of Excel's CHOOSE function?",
                "How to free c++ memory vector<int> * arr?"]
    answers = ["sql server equivalent excels choose function", 
               "free c++ memory vectorint arr"]
    for ex, ans in zip(examples, answers):
        if text_prepare(ex) != ans:
            return "Wrong answer for the case: '%s'" % ex
    return 'Basic tests are passed.'

print(test_text_prepare())

X_train = [text_prepare(x) for x in X_train]
X_val = [text_prepare(x) for x in X_val]
X_test = [text_prepare(x) for x in X_test]

X_train[:3]


# Dictionary of all tags from train corpus with their counts.
tags_counts = {}
# Dictionary of all words from train corpus with their counts.
words_counts = {}
######################################
######### YOUR CODE HERE #############
######################################
from collections import Counter
tags_counts = Counter([tag for taglist in y_train for tag in taglist])
words_counts = Counter([word for question in X_train for word in question.split(' ')])

most_common_tags = sorted(tags_counts.items(), key=lambda x: x[1], reverse=True)[:3]
most_common_words = sorted(words_counts.items(), key=lambda x: x[1], reverse=True)[:3]

#%%
DICT_SIZE = 5000
####### YOUR CODE HERE #######
WORDS_TO_INDEX = {word[0]:int_idx for int_idx, word in enumerate(sorted(words_counts.items(), key=lambda x: x[1], reverse=True)[:DICT_SIZE])}
####### YOUR CODE HERE #######
INDEX_TO_WORDS = dict(zip(WORDS_TO_INDEX.values(),WORDS_TO_INDEX.keys()))
ALL_WORDS = WORDS_TO_INDEX.keys()

def my_bag_of_words(text, words_to_index, dict_size):
    """
        text: a string
        dict_size: size of the dictionary
        
        return a vector which is a bag-of-words representation of 'text'
    """
    result_vector = np.zeros(dict_size)
    ######################################
    ######### YOUR CODE HERE #############
    ######################################
    text_tokenized = text.split(' ')
    for token in text_tokenized:
        if token in words_to_index:
            result_vector[words_to_index[token]] += 1
    return result_vector

def test_my_bag_of_words():
    words_to_index = {'hi': 0, 'you': 1, 'me': 2, 'are': 3}
    examples = ['hi how are you']
    answers = [[1, 1, 0, 1]]
    for ex, ans in zip(examples, answers):
        if (my_bag_of_words(ex, words_to_index, 4) != ans).any():
            return "Wrong answer for the case: '%s'" % ex
    return 'Basic tests are passed.'

print(test_my_bag_of_words())

from scipy import sparse as sp_sparse

X_train_mybag = sp_sparse.vstack([sp_sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in X_train])
X_val_mybag = sp_sparse.vstack([sp_sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in X_val])
X_test_mybag = sp_sparse.vstack([sp_sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in X_test])
print('X_train shape ', X_train_mybag.shape)
print('X_val shape ', X_val_mybag.shape)
print('X_test shape ', X_test_mybag.shape)

row = X_train_mybag[10].toarray()[0]
non_zero_elements_count = row.reshape(-1,1).sum(axis=0)

#%%
from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf_features(X_train, X_val, X_test):
    """
        X_train, X_val, X_test — samples        
        return TF-IDF vectorized representation of each sample and vocabulary
    """
    # Create TF-IDF vectorizer with a proper parameters choice
    # Fit the vectorizer on the train set
    # Transform the train, test, and val sets and return the result
    
    ####### YOUR CODE HERE #######
    tfidf_vectorizer = TfidfVectorizer(analyzer='word', token_pattern = '(\S+)', min_df = 5, max_df = 0.9, ngram_range=(1,2))
    
    ######################################
    ######### YOUR CODE HERE #############
    ######################################
    X_train = tfidf_vectorizer.fit_transform(X_train)
    X_val = tfidf_vectorizer.transform(X_val)
    X_test = tfidf_vectorizer.transform(X_test)
    
    return X_train, X_val, X_test, tfidf_vectorizer.vocabulary_

X_train_tfidf, X_val_tfidf, X_test_tfidf, tfidf_vocab = tfidf_features(X_train, X_val, X_test)
tfidf_reversed_vocab = {i:word for word,i in tfidf_vocab.items()}

all([tag in tfidf_vocab for tag in ['c++','c#']])
#%%
                                    
from sklearn.preprocessing import MultiLabelBinarizer                                   

mlb = MultiLabelBinarizer(classes=sorted(tags_counts.keys()))
y_train = mlb.fit_transform(y_train)
y_val = mlb.fit_transform(y_val)

from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

def train_classifier(X_train, y_train):
    """
      X_train, y_train — training data
      
      return: trained classifier
    """
    
    # Create and fit LogisticRegression wraped into OneVsRestClassifier.

    ######################################
    ######### YOUR CODE HERE #############
    ######################################
    clf = OneVsRestClassifier(LogisticRegression())
    clf.fit(X_train,y_train)
    
    return clf

classifier_mybag = train_classifier(X_train_mybag, y_train)
classifier_tfidf = train_classifier(X_train_tfidf, y_train)

y_val_predicted_labels_mybag = classifier_mybag.predict(X_val_mybag)
y_val_predicted_scores_mybag = classifier_mybag.decision_function(X_val_mybag)

y_val_predicted_labels_tfidf = classifier_tfidf.predict(X_val_tfidf)
y_val_predicted_scores_tfidf = classifier_tfidf.decision_function(X_val_tfidf)

y_val_pred_inversed = mlb.inverse_transform(y_val_predicted_labels_tfidf)
y_val_inversed = mlb.inverse_transform(y_val)
for i in range(3):
    print('Title:\t{}\nTrue labels:\t{}\nPredicted labels:\t{}\n\n'.format(
        X_val[i],
        ','.join(y_val_inversed[i]),
        ','.join(y_val_pred_inversed[i])
    ))
#%%    

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score

def print_evaluation_scores(y_val, predicted):
    
    ######################################
    ######### YOUR CODE HERE #############
    ######################################
    
    # Accuracy
    print('Acuracy score: {:.2f}'.format(accuracy_score(y_val, predicted)))
    # F1-Score
    print('F1-Score macro-average: {:.2f}'.format(f1_score(y_val,predicted,average='macro')))
    print('F1-Score micro-average: {:.2f}'.format(f1_score(y_val,predicted,average='micro')))
    print('F1-Score weighted-average: {:.2f}'.format(f1_score(y_val,predicted,average='weighted')))
    # Precision
    print('Precision macro-average: {:.2f}'.format(average_precision_score(y_val,predicted,average='macro')))
    print('Precision micro-average: {:.2f}'.format(average_precision_score(y_val,predicted,average='micro')))
    print('Precision weighted-average: {:.2f}'.format(average_precision_score(y_val,predicted,average='weighted')))
    
print('Bag-of-words')
print_evaluation_scores(y_val, y_val_predicted_labels_mybag)
print('Tfidf')
print_evaluation_scores(y_val, y_val_predicted_labels_tfidf)

from metrics import roc_auc
n_classes = len(tags_counts)
roc_auc(y_val, y_val_predicted_scores_mybag, n_classes)

n_classes = len(tags_counts)
roc_auc(y_val, y_val_predicted_scores_tfidf, n_classes)
#%%



######################################
######### YOUR CODE HERE #############
######################################
def grid_search(c,penalty,X_train_mybag,X_train_tfidf,X_val_mybag,X_val_tfidf,y_train,y_val):
    
    # Define model
    clf = OneVsRestClassifier(LogisticRegression(C = c, penalty = penalty))
    
    # Fit, train and predict mybag
    clf_mybag = clf.fit(X_train_mybag,y_train)
    y_val_predicted_labels_mybag = clf_mybag.predict(X_val_mybag)
 
    # Fit, train and predict tfidf
    clf_tfidf = clf.fit(X_train_tfidf,y_train)
    y_val_predicted_labels_tfidf = clf_tfidf.predict(X_val_tfidf)
    
    # Evaluate
    print("For parameters C={}, penalty={}".format(c,penalty))
    print("mybag: F1-score = {:.2f}".format(f1_score(y_val, y_val_predicted_labels_mybag, average = 'weighted'
)))
    print("tfidf: F1-score = {:.2f}".format(f1_score(y_val, y_val_predicted_labels_tfidf, average = 'weighted'
)))


    
from itertools import product
C = [0.1,1,10,100]
regularization = ['l1','l2']
grid = list(product(C,regularization))

for c,reg in grid:
    grid_search(c,reg,X_train_mybag,X_train_tfidf,X_val_mybag,X_val_tfidf,y_train,y_val)
#%%


######### YOUR CODE HERE #############
clf = OneVsRestClassifier(LogisticRegression(C = 1, penalty = 'l1'))
clf_mybag = clf.fit(X_train_mybag,y_train)
test_predictions = clf_mybag.predict(X_test_mybag)
test_pred_inversed = mlb.inverse_transform(test_predictions)

test_predictions_for_submission = '\n'.join('%i\t%s' % (i, ','.join(row)) for i, row in enumerate(test_pred_inversed))


def print_words_for_tag(classifier, tag, tags_classes, index_to_words, all_words):
    """
        classifier: trained classifier
        tag: particular tag
        tags_classes: a list of classes names from MultiLabelBinarizer
        index_to_words: index_to_words transformation
        all_words: all words in the dictionary
        
        return nothing, just print top 5 positive and top 5 negative words for current tag
    """
    print('Tag:\t{}'.format(tag))
    
    # Extract an estimator from the classifier for the given tag.
    # Extract feature coefficients from the estimator. 
    
    ######################################
    ######### YOUR CODE HERE #############
    ######################################
    
    # top-5 words sorted by the coefficiens.
    top_positive_words = 
    # bottom-5 words  sorted by the coefficients.
    top_negative_words = 
    print('Top positive words:\t{}'.format(', '.join(top_positive_words)))
    print('Top negative words:\t{}\n'.format(', '.join(top_negative_words)))
                                    

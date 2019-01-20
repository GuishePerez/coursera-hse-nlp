# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 19:06:21 2019

@author: Guishe
"""

#%%
#import sys
#sys.path.append("..")
#from common.download_utils import download_week2_resources
#
#download_week2_resources()
#%%
def read_data(file_path):
    tokens = []
    tags = []
    
    tweet_tokens = []
    tweet_tags = []
    for line in open(file_path, encoding='utf-8'):
        line = line.strip()
        if not line:
            if tweet_tokens:
                tokens.append(tweet_tokens)
                tags.append(tweet_tags)
            tweet_tokens = []
            tweet_tags = []
        else:
            token, tag = line.split()
            # Replace all urls with <URL> token
            # Replace all users with <USR> token

            ######################################
            ######### YOUR CODE HERE #############
            ######################################
            if token.startswith('@'):
                token = '<USR>'
            if token.lower().startswith('http://') or token.lower().startswith('https://'):
                token = '<URL>'           
            
            tweet_tokens.append(token)
            tweet_tags.append(tag)
            
    return tokens, tags

train_tokens, train_tags = read_data('data/train.txt')
validation_tokens, validation_tags = read_data('data/validation.txt')
test_tokens, test_tags = read_data('data/test.txt')

#%%
from collections import defaultdict

def build_dict(tokens_or_tags, special_tokens):
    """
        tokens_or_tags: a list of lists of tokens or tags
        special_tokens: some special tokens
    """
    # Create a dictionary with default value 0
    tok2idx = defaultdict(lambda: 0)
    idx2tok = []
    
    # Create mappings from tokens (or tags) to indices and vice versa.
    # At first, add special tokens (or tags) to the dictionaries.
    # The first special token must have index 0.
    
    # Mapping tok2idx should contain each token or tag only once. 
    # To do so, you should:
    # 1. extract unique tokens/tags from the tokens_or_tags variable, which is not
    #    occur in special_tokens (because they could have non-empty intersection)
    # 2. index them (for example, you can add them into the list idx2tok
    # 3. for each token/tag save the index into tok2idx).
    
    ######################################
    ######### YOUR CODE HERE #############
    ######################################
    tokens = list(set([token for tweet in tokens_or_tags for token in tweet]))
    vocab = special_tokens + tokens
    
    for i,token in enumerate(vocab):
        tok2idx[token] = i
        idx2tok.append(token)
    
    return tok2idx, idx2tok

special_tokens = ['<UNK>', '<PAD>']
special_tags = ['O']

# Create dictionaries 
token2idx, idx2token = build_dict(train_tokens + validation_tokens, special_tokens)
tag2idx, idx2tag = build_dict(train_tags, special_tags)

#%%





# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 20:44:40 2019

@author: Guishe
"""


def sentence_to_ids(sentence, word2id, padded_len):
    """ Converts a sequence of symbols to a padded sequence of their ids.
    
      sentence: a string, input/output sequence of symbols.
      word2id: a dict, a mapping from original symbols to ids.
      padded_len: an integer, a desirable length of the sequence.

      result: a tuple of (a list of ids, an actual length of sentence).
    """
    ######### YOUR CODE HERE #############
    eq_len = len(sentence)
    delta_len = padded_len - eq_len
    if delta_len < 0:
        sent_ids = [word2id[sentence[i]] if i < eq_len - 1 else word2id[end_symbol] for i in range(padded_len)]
        sent_len = len(sent_ids)
        sent_ids.append(word2id[padding_symbol] for _ in range(delta_len))
    elif delta_len < 2:
        sent_ids = [word2id[sentence[i]] if i < padded_len - 1 else word2id[end_symbol] for i in range(padded_len)]
        sent_len = len(sent_ids)
    else:
        sent_ids = [word2id[sentence[i]] for i in range(eq_len)]
        sent_ids.append(word2id[end_symbol])
        sent_len = len(sent_ids)
        for _ in range(delta_len-1):
            sent_ids.append(word2id[padding_symbol])
    
    return sent_ids, sent_len


word2id = {symbol:i for i, symbol in enumerate('#^$+-1234567890')}
id2word = {i:symbol for symbol, i in word2id.items()}

start_symbol = '^'
end_symbol = '$'
padding_symbol = '#'

def test_sentence_to_ids():
    sentences = [("123+123", 7), ("123+123", 8), ("123+123", 10)]
    expected_output = [([5, 6, 7, 3, 5, 6, 2], 7), 
                       ([5, 6, 7, 3, 5, 6, 7, 2], 8), 
                       ([5, 6, 7, 3, 5, 6, 7, 2, 0, 0], 8)] 
    for (sentence, padded_len), (sentence_ids, expected_length) in zip(sentences, expected_output):
        output, length = sentence_to_ids(sentence, word2id, padded_len)
        if output != sentence_ids:
            return("Convertion of '{}' for padded_len={} to {} is incorrect.".format(
                sentence, padded_len, output))
        if length != expected_length:
            return("Convertion of '{}' for padded_len={} has incorrect actual length {}.".format(
                sentence, padded_len, length))
    return("Tests passed.")
    
    
    
print(test_sentence_to_ids())









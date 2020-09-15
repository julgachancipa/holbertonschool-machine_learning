#!/usr/bin/env python3
"""
0. Bag Of Words
"""
from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """
    creates a bag of words embedding matrix
    :param sentences: is a list of sentences to analyze
    :param vocab: is a list of the vocabulary words to use for
    the analysis
        If None, all words within sentences should be used
    :return: embeddings, features
        embeddings is a numpy.ndarray of shape (s, f) containing
        the embeddings
            s is the number of sentences in sentences
            f is the number of features analyzed
        features is a list of the features used for embeddings
    """
    vectorizer = CountVectorizer(vocabulary=vocab)
    X = vectorizer.fit_transform(sentences)
    embeddings = X.toarray()
    features = vectorizer.get_feature_names()
    return embeddings, features

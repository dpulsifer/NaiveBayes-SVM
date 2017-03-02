import os
import string
import nltk
import pandas as pd
import numpy as np

from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import punkt
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = {}
snowball_stemmer = SnowballStemmer("english")

def text_stemmer(tokens, stemmer):
    words_stem = []
    for word in tokens:
        words_stem.append(snowball_stemmer.stem(word))
    return words_stem

def text_tokenizer(document):
    doc_tokens = nltk.word_tokenize(document)
    stemmed_tokens = text_stemmer(doc_tokens, snowball_stemmer)
    return stemmed_tokens

def top_tfidf_features(row, features):
    #get top n tf_idf values per row
    top_ids = np.argsort(row)[::-1][:20]
    top_features = [(features[i], row[i]) for i in top_ids]
    df = pd.DataFrame(top_features)
    df.columns = ['feature', 'tfidf']
    return df

def top_mean_features(tfidf_matrix, feature_names):
    #return top features with highest mean weight in corpus
    D = tfidf_matrix.toarray()

    tfidf_means = np.mean(D, axis=0)
    return top_tfidf_features(tfidf_means, feature_names)

table_punc = str.maketrans({key: None for key in string.punctuation})
table_nums = str.maketrans({key: None for key in string.digits})

for subdir, dirs, files in os.walk('newsgroups'):
     for file in files:
        if (subdir == 'newsgroups/alt.atheism') or (subdir == 'newsgroups/sci.space') or (subdir == 'newsgroups/comp.graphics') or (subdir == 'newsgroups/talk.religion.misc'):
            if file == '.DS_Store':
                continue
            file_path =  subdir + '/' + file
            doc_text =  open(file_path, encoding = "ISO-8859-1").read()
            text_lower = doc_text.lower()
            text_nopunc = text_lower.translate(table_punc)
            text_nonums = text_nopunc.translate(table_nums)
            corpus[file] = text_nonums

tfidf_vectorizer_unigram = TfidfVectorizer( tokenizer=text_tokenizer, stop_words='english' )
tfidf_vectorizer_bigram = TfidfVectorizer( tokenizer=text_tokenizer, ngram_range = (2,2), stop_words='english' )

tfidf_matrix_unigram = tfidf_vectorizer_unigram.fit_transform(corpus.values())
tfidf_matrix_bigram = tfidf_vectorizer_bigram.fit_transform(corpus.values())

feature_names_unigram = tfidf_vectorizer_unigram.get_feature_names()
feature_names_bigram = tfidf_vectorizer_bigram.get_feature_names()

top_terms_unigram = top_mean_features(tfidf_matrix_unigram, feature_names_unigram)
top_terms_bigram = top_mean_features(tfidf_matrix_bigram, feature_names_bigram)

print ("Top Unigrams: ")
print(top_terms_unigram)
print()
print("Top Bigrams:")
print(top_terms_bigram)

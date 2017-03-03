import os
import string
import nltk
import pandas as pd
import numpy as np

from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import punkt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn import svm

corpus = {}
categories = []
for i in range(160):
    if (i < 40 ): categories.append('1')
    if (i >= 40 and i < 80): categories.append('2')
    if (i >= 80 and i < 120): categories.append('3')
    if (i >= 120 and i < 160): categories.append('4')

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

#create tfidf matrix
tfidf_vectorizer_unigram = TfidfVectorizer( tokenizer=text_tokenizer, stop_words='english' )
tfidf_matrix_unigram = tfidf_vectorizer_unigram.fit_transform(corpus.values())

#separate training and test data
train_data, test_data, train_category, test_category = train_test_split(tfidf_matrix_unigram, categories, test_size = 0.3)

#create and train naive bayes classifier, and run test data
nb_classifier = MultinomialNB().fit(train_data, train_category)
nb_predicted = nb_classifier.predict(test_data)

#create and train svm classifier and run test data
svm_classifier = svm.LinearSVC()
svm_classifier.fit(train_data, train_category)
svm_predicted = svm_classifier.predict(test_data)

#plot confusion matrices
nb_conf_matrix = confusion_matrix(test_category, nb_predicted)
svm_conf_matrix = confusion_matrix(test_category, svm_predicted)

print ("Unigram")
print ("Naive Bayes Confusion Matrix:")
print (nb_conf_matrix)
print ("SVM Confusion Matrix:")
print (svm_conf_matrix)

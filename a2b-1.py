
# coding: utf-8

# In[17]:

import os
import string
import nltk

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
        
tfidf_vectorizer = TfidfVectorizer( tokenizer=text_tokenizer, ngram_range = (2,2), stop_words='english' )
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus.values())

tfidf_matrix

matrix_array = tfidf_matrix.toarray()

feature_names = tfidf_vectorizer.get_feature_names()

print ([(feature_names[x],l[x]) for x in (l*-1).argsort()][:20])
        


# In[ ]:




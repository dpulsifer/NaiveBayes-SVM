library(datasets)
library(tm)
library(SnowballC)
library(RWeka)
library(NLP)

options(mc.cores=1)

setwd("/Users/duncanpulsifer/Documents/Dalhousie/CSCI\ 3151/Assignments/a2")

corpus <- Corpus(DirSource("NewsGroups", encoding = "UTF-8", recursive = TRUE), readerControl = list(language = "en"))

#preprocess corpus
corpus <- tm_map (corpus, content_transformer(tolower))
corpus <- tm_map (corpus, removeNumbers)
corpus <- tm_map (corpus, removePunctuation)
corpus <- tm_map (corpus, removeWords, stopwords("english"))
corpus <- tm_map (corpus, stemDocument)

#bigram tokenizer
bigramTokenizer <- BigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 2, max = 2))

#create unigram DTM, remove sparse terms, and print top 20 unigrams
dtmUnigram <- DocumentTermMatrix(corpus, control = list(weighting = function(x) weightTfIdf(x, normalize = TRUE)))
print(dtmUnigram)
dtmUnigram <- removeSparseTerms(dtmUnigram, 0.9)
unigramTopTerms <- (sort(colSums(as.matrix(dtmUnigram)/160), decreasing = TRUE)[1:20])
print(unigramTopTerms)

#create bigram DTM, remove sparse terms, and print top 20 bigrams
dtmBigram <- DocumentTermMatrix(corpus, control = list(tokenize = bigramTokenizer, weighting = function(x) weightTfIdf(x, normalize = TRUE)))
print(dtmBigram)
dtmBigram <- removeSparseTerms(dtmBigram, 0.875)
bigramTopTerms <- (sort(colSums(as.matrix(dtmBigram)/160), decreasing = TRUE)[1:20])
print(bigramTopTerms)

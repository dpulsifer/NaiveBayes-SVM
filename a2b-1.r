library(datasets)
library(tm)
library(SnowballC)
library(RWeka)
library(NLP)

options(mc.cores=1)

setwd("/Users/duncanpulsifer/Documents/Dalhousie/CSCI\ 3151/Assignments/a2")

cryptSpaceCorpus <- Corpus(DirSource("NewsGroups", encoding = "UTF-8", recursive = TRUE), readerControl = list(language = "en"))

#preprocess corpus
cryptSpaceCorpus <- tm_map (cryptSpaceCorpus, content_transformer(tolower))
cryptSpaceCorpus <- tm_map (cryptSpaceCorpus, removeNumbers)
cryptSpaceCorpus <- tm_map (cryptSpaceCorpus, removePunctuation)
cryptSpaceCorpus <- tm_map (cryptSpaceCorpus, removeWords, stopwords("english"))
cryptSpaceCorpus <- tm_map (cryptSpaceCorpus, stemDocument)

#bigram tokenizer
bigramTokenizer <- BigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 2, max = 2))

#create unigram DTM, remove sparse terms, and print top 20 unigrams
dtmUnigram <- DocumentTermMatrix(cryptSpaceCorpus, control = list(weighting = function(x) weightTfIdf(x, normalize = TRUE)))
print(dtmUnigram)
dtmUnigram <- removeSparseTerms(dtmUnigram, 0.9)
unigramTopTerms <- (sort(colSums(as.matrix(dtmUnigram)/160), decreasing = TRUE)[1:20])
print(unigramTopTerms)

#create bigram DTM, remove sparse terms, and print top 20 bigrams
dtmBigram <- DocumentTermMatrix(cryptSpaceCorpus, control = list(tokenize = bigramTokenizer, weighting = function(x) weightTfIdf(x, normalize = TRUE)))
print(dtmBigram)
dtmBigram <- removeSparseTerms(dtmBigram, 0.875)
bigramTopTerms <- (sort(colSums(as.matrix(dtmBigram)/160), decreasing = TRUE)[1:20])
print(bigramTopTerms)

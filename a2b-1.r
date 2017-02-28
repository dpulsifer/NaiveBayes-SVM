library(datasets)
library(tm)
library(SnowballC)
library(RWeka)
library(NLP)

options(mc.cores=1)

setwd("/Users/duncanpulsifer/Documents/Dalhousie/CSCI\ 3151/Assignments/a2")

cryptSpaceCorpus <- Corpus(DirSource("NewsGroups", encoding = "UTF-8", recursive = TRUE), readerControl = list(language = "en"))

cryptSpaceCorpus <- tm_map (cryptSpaceCorpus, content_transformer(tolower))
cryptSpaceCorpus <- tm_map (cryptSpaceCorpus, removeNumbers)
cryptSpaceCorpus <- tm_map (cryptSpaceCorpus, removePunctuation)
cryptSpaceCorpus <- tm_map (cryptSpaceCorpus, removeWords, stopwords("english"))
cryptSpaceCorpus <- tm_map (cryptSpaceCorpus, stemDocument)

bigramTokenizer <- BigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 2, max = 2))

dtmUnigram <- DocumentTermMatrix(cryptSpaceCorpus, control = list(weighting = function(x) weightTfIdf(x, normalize = TRUE)))

dtmBigram <- DocumentTermMatrix(cryptSpaceCorpus, control = list(tokenize = bigramTokenizer, weighting = function(x) weightTfIdf(x, normalize = TRUE)))

dtmBigram <- removeSparseTerms(dtmBigram, 0.8)

print(dtmUnigram)

print(dtmBigram)

inspect(dtmBigram)

library(datasets)
library(tm)
library(SnowballC)
library(RWeka)
library(NLP)
library(e1071)
library(Matrix)
library(class)

options(mc.cores=1)

setwd("/Users/duncanpulsifer/Documents/Dalhousie/CSCI\ 3151/Assignments/a2")

corpus <- Corpus(DirSource("NewsGroups", encoding = "UTF-8", recursive = TRUE),		readerControl = list(language = "en"))

#preprocess corpus
corpus <- tm_map (corpus, content_transformer(tolower))
corpus <- tm_map (corpus, removeNumbers)
corpus <- tm_map (corpus, removePunctuation)
corpus <- tm_map (corpus, removeWords, stopwords("english"))
corpus <- tm_map (corpus, stemDocument)

#bigram tokenizer
bigramTokenizer <- BigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 2, max = 2))

#convert to tf-idf weighted document term matrix
dtmBigram <- DocumentTermMatrix(corpus, control = list(tokenize = bigramTokenizer,weighting = function(x) weightTfIdf(x, normalize = TRUE)))

dtmBigram <- removeSparseTerms(dtmBigram, 0.99)

#convert to matrix and then dataframe
bigramModel <- as.matrix(dtmBigram)
bigramModel <- as.data.frame(bigramModel)

#append training column and assign 1 or 0 in 70/30 proportion
bigramModel[,"training"] <- ifelse(runif(nrow(bigramModel))<0.70,1,0)

#add category labels to rows
bigramModel$Category <- NA
for ( i in 1:nrow(bigramModel) ) {
	if ( 1 <= i & i <= 40 ) { bigramModel$Category[i] <- 1 }
	if ( 40 < i & i <= 80 ) { bigramModel$Category[i] <- 2 }
	if ( 80 < i & i <= 120 ) { bigramModel$Category[i] <- 3 }
	if ( 120 < i & i <= 160 ) { bigramModel$Category[i] <- 4 }
}

bigramModel$Category <- as.factor(bigramModel$Category)

#get number of training indication column for set division
trainingColumnNumber <- grep("training", names(bigramModel))

#divide training and testing sets
trainingBigramModel <- bigramModel[bigramModel$training==1,-trainingColumnNumber]
testingBigramModel <- bigramModel[bigramModel$training==0,-trainingColumnNumber]

#get number of training indication column
categoryColumnNumber <- grep("Category", names(testingBigramModel))

#run naive bayes and svm algorithms
bigramNB <- naiveBayes(Category~., data = trainingBigramModel)
bigramSVM <- svm(Category~., data = trainingBigramModel, kernel='linear', cost = 62.5)

#run predictions with testing data
bigramNB_Test <- predict(bigramNB, testingBigramModel[,-categoryColumnNumber], na.action = na.pass)
bigramSVM_Test <- predict(bigramSVM, testingBigramModel[,-categoryColumnNumber], na.action = na.pass)

#generate tables
confusion_matrix <- table(pred=bigramNB_Test, true=testingBigramModel$Category)
svm_accuracies <- table(pred=bigramSVM_Test, true=testingBigramModel$Category)

print("Bigram Naive Bayes Output:")
print(confusion_matrix)
print("Bigram SVM Output:")
print(svm_accuracies)

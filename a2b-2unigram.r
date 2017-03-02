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

#convert to tf-idf weighted document term matrix
dtmUnigram <- DocumentTermMatrix(corpus, control = list(weighting = function(x) weightTfIdf(x, normalize = TRUE)))

dtmUnigram <- removeSparseTerms(dtmUnigram, 0.995)

#convert to matrix and then dataframe
unigramModel <- as.matrix(dtmUnigram)
unigramModel <- as.data.frame(unigramModel)

#append training column and assign 1 or 0 in 70/30 proportion
unigramModel[,"training"] <- ifelse(runif(nrow(unigramModel))<0.70,1,0)

#add category labels to rows
unigramModel$Category <- NA
for ( i in 1:nrow(unigramModel) ) {
	if ( 1 <= i & i <= 40 ) { unigramModel$Category[i] <- 1 }
	if ( 40 < i & i <= 80 ) { unigramModel$Category[i] <- 2 }
	if ( 80 < i & i <= 120 ) { unigramModel$Category[i] <- 3 }
	if ( 120 < i & i <= 160 ) { unigramModel$Category[i] <- 4 }
}

#get number of training indication column for set division
trainingColumnNumber <- grep("training", names(unigramModel))

#divide training and testing sets
trainingUnigramModel <- unigramModel[unigramModel$training==1,-trainingColumnNumber]
testingUnigramModel <- unigramModel[unigramModel$training==0,-trainingColumnNumber]

#get number of training indication column
categoryColumnNumber <- grep("Category", names(testingUnigramModel))

#run naive bayes and svm algorithms
unigramNB <- naiveBayes(as.factor(Category)~., data = trainingUnigramModel)
unigramSVM <- svm(as.factor(Category)~., data = trainingUnigramModel)

#run predictions with testing data
unigramNB_Test <- predict(unigramNB, testingUnigramModel[,-categoryColumnNumber], na.action = na.pass)
unigramSVM_Test <- predict(unigramSVM, testingUnigramModel[,-categoryColumnNumber], na.action = na.pass)

#generate tables
confusion_matrix <- table(pred=unigramNB_Test, true=testingUnigramModel$Category)
svm_accuracies <- table(pred=unigramSVM_Test, true=testingUnigramModel$Category)

print(confusion_matrix)
print(svm_accuracies)

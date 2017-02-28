# Q4 (80). Naıve Bayes and SVM in Python and R

In this question you will practice machine learning algorithms namely Multinomial Naiıve Bayes and SVM. You will also practice making confusion matrix and calculating accuracy of predictions.
Deliverables:
* A concise report in pdf format that includes your results, accuracies, discussion and any other insights that you might have by tuning any algorithm parameters for the tasks defined below.
* Your runnable python and R source code (in both languages)

Submissions:
Please upload your completed assignment as a single zip file. The filename MUST include last name and your banner number(e.g.A1 Sharifirad B001111.zip)

Dataset:
Please download 20-newgroup dataset from [Newsgroups] (http://qwone.com/~jason/20Newsgroups/). Download the last .tar.gz file(20news-18828.tar.gz ). From the 20 groups just consider ’alt.atheism’, ’talk.religion.misc’, ’comp.graphics’,’sci.space’,). Please order the text file in each category based on their size and in each category choose the top 40 text files with highest size). If you need to consider a label for each category, label them as following (’alt.atheism’=1, ’talk.religion.misc’=2, ’comp.graphics’=3, ’sci.space’=4).
Useful python and R packages:
* Scipy/numpy: useful for multidimensional arrays, vector and matrix operations.
* Scikit-learn: machine learning library for tasks including classification, regression, clustering, feature selection.
* Nltk: NLTK is a leading platform for building Python programs to work with human language data. It provides easy-to-use interfaces to over 50 corpora and lexical resources such as WordNet, along with a suite of text processing libraries for classification, tokenization, stemming, tagging, parsing, and semantic reasoning.

Your tasks:

1. Preprocess the data: stop word removal, stemming, remove numbers if necessary and tokenize it (use suitable libraries for these tasks). Then, try to make two forms from your original dataset, unigram and bigram, run tfidf on each of the grams and rank the top features (words or pair words). Report the top 20 features by their value in each dataset.
2. Split your data (unigram and bigram separately) randomly into a training and testing set (e.g. 70%-30%). Train MultinomialNB (Multinomial Nave Bayes) and SVM using the training data. Then run your test data. Report the confusion matrix and accuracy for each of the two algorithms. Explain how confusion matrix is used for accuracy. which algorithm has higher accuracy on each dataset and why? Note: for the purpose of this assignment, you will have:
* two datasets in total (unigram and bigram)
* four confusion matrixes (two algorithms and two datasets)
* two accuracies

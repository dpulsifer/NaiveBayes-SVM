## Naıve Bayes and SVM in Python and R

Implementations of Naive Bayes and Support Vector Machine (SVM) in Python and R. 20-newgroup dataset from [Newsgroups] (http://qwone.com/~jason/20Newsgroups/) was used. Only documents from ’alt.atheism’, ’talk.religion.misc’, ’comp.graphics’, and ’sci.space’ were considered, and from these, only the largest 40 files were used.

### Part 1

The A2b-1 files print the top 20 unigram and bigram terms from the corpus as a whole. The top terms were determined by mean tf-idf weight across all documents. Deviating results in the Python and R implementations are the result of different packages and algorithms used in each respective language. 

### Part 2

The A2b-2 files print Naive Bayes and SVM confusion matrices for unigrams and bigrams in both Python and R. The dataset was split 70-30 into training and test sets. Results were wide ranging, though the SVM implementations generally produced better accuracy. These programs only run the algorithms one time, and thus do not implement n-fold cross validation. An n-fold cross validation takes the mean accuracy of n trials to provide a more reliable overall accuracy.

### Notable R Libraries Used

* tm
* SnowballC
* RWeka
* NLP
* e1071

### Notable Python Libraries Used

* nltk
* sklearn
* pandas
* numpy

### To run R files

* You must change the location of the working directory within the R files to the location of the NewsGroups datasets.
* Then, from the command line, enter R to start the R environment. (Note: it may be beneficial to run R --max-pp-size=500000 to increase the size of your environment and avoid stack overflow.)
* From the command line, enter: source(/directory_of_file_to_run/filename.r)
* Results will print to console (it can take some time for the program to output results).

### To run Python files.

* From the command line, navigate to directory of files.
* From the command line, enter: python3 filename.py
* Results will print to console.




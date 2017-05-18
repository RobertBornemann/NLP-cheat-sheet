# natural language processing

# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter='\t', quoting=3)

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')

corpus = []

for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    porter_stemmer = PorterStemmer()
    review =  [porter_stemmer.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review =  ' '.join(review)
    corpus.append(review)

# Creating the bags of words model

from sklearn.feature_extraction.text import CountVectorizer
count_vectorizer = CountVectorizer(max_features=1500)

X = count_vectorizer.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)







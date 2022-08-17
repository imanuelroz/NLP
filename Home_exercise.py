import re
import string

import gensim
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

pd.set_option('display.max_columns', 4)
pd.set_option('display.max_colwidth', 100)
data = pd.read_csv('Data/language_detection.csv')
print(data.head())

#see if there is class imbalance

print(data["language"].value_counts())
#we need to encode the label values
X = data["text"]
print(type(X[0]))

y = data["language"]

labels = LabelEncoder()
y = labels.fit_transform(y)
print(y)

# creating a list for appending the preprocessed text
data_list = []
# iterating through all the text
for text in X:
       # removing the symbols and numbers
        text = re.sub(r'[!@#$(),n"%^*?:;~`0-9]', ' ', text)
        text = re.sub(r'[[]]', ' ', text)
        # converting the text to lower case
        text = text.lower()
        # appending to data_list
        data_list.append(text)


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(analyzer = 'char', ngram_range=(2,2))
X = cv.fit_transform(data_list).toarray()
print(X.shape) # (10337, 8849)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

print(X_train[1])


#model = MultinomialNB()
model1 = LogisticRegression(solver='lbfgs', max_iter=50)

#model.fit(X_train, y_train)
model1.fit(X_train, y_train)
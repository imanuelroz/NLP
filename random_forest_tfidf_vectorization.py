import nltk
import re
import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split

stopwords = nltk.corpus.stopwords.words('english')

messages = pd.read_csv('Data/spam.csv', encoding='latin-1')
messages = messages.drop(labels=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
messages.columns = ["label", "text"]


def clean_text(text):
    text = "".join([char.lower() for char in text if char not in string.punctuation])
    tokens = re.split('\W+', text)
    text = [word for word in tokens if word not in stopwords]
    return text

tfidf_vect = TfidfVectorizer(analyzer=clean_text)
X_tfidf = tfidf_vect.fit_transform(messages['text'])


X_features = pd.DataFrame(X_tfidf.toarray())
print(X_features.head())

# Split data into training and test sets, we will evaluate RandomForest on the test sets
#If you want to use also the validation set then apply again the train_test_split on the X_train, X_valid  etc etc
X_train, X_test, y_train, y_test = train_test_split(X_features,
                                                    messages['label'],
                                                    test_size=0.2,
                                                    random_state=12)

print(X_test)


# Fit a basic Random Forest model
rf = RandomForestClassifier() #instantiate the default model
rf_model = rf.fit(X_train, y_train)

# Make predictions on the test set using the fit model
y_pred = rf_model.predict(X_test)

#Evaluate model predictions using precision and recall

precision = precision_score(y_test, y_pred, pos_label='spam') #spam is the positive label we are trying to predict
recall = recall_score(y_test, y_pred, pos_label='spam')
print('Precision: {} / Recall: {}'.format(round(precision, 3), round(recall, 3)))

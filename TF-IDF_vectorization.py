import pandas as pd
import re
import string
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

pd.set_option('display.max_colwidth', 100)

stopwords = nltk.corpus.stopwords.words('english')

messages = pd.read_csv('Data/spam.csv', encoding='latin-1')
messages = messages.drop(labels=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis = 1)
messages.columns = ["label", "text"]



#before we vectorize we have to clean de data

def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    print(text)
    tokens = re.split('\W+', text)
    text = [word for word in tokens if word not in stopwords]
    return text

tfidf_vect = TfidfVectorizer(analyzer=clean_text) #so first it applies clean_text and then on it the tfidf

# Fit a basic TFIDF Vectorizer and view the results
X_tfidf = tfidf_vect.fit_transform(messages['text'])
print(X_tfidf.shape)
print(tfidf_vect.get_feature_names)#return all of the words that are vectorized or learned from our training data



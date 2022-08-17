# Read in data, clean it, and then split into train and test sets
import gensim
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.max_columns', 4)

messages = pd.read_csv('Data/spam.csv', encoding='latin-1')
messages = messages.drop(labels=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
messages.columns = ["label", "text"]
messages['text_clean'] = messages['text'].apply(lambda x: gensim.utils.simple_preprocess(x))
#print(messages.head())

X_train, X_test, y_train, y_test = train_test_split(messages['text_clean'],
                                                    messages['label'],
                                                    test_size=0.2, random_state=13)
print(type(X_train[0])) #remember each row is a list

#doc2vec expects tagged documents, a simple way to use a tag is to use the index as the tag
#create tagged document objects to prepare to train the model

tagged_docs = [gensim.models.doc2vec.TaggedDocument(v, [i]) for i, v in enumerate(X_train)]
#note that the index has to be passed as a second argument inside of a list
print(tagged_docs[0])

#now we are ready to train doc2vec model on our training data

d2v_model = gensim.models.Doc2Vec(tagged_docs,
                                  vector_size=100,
                                  window=5,
                                  min_count=2)
#What happens if we pass in a single worl like we did for word2vec? it will raise an error
#d2v_model.infer_vector('text') ---> ERROR

# What happens if we pass in a list of words?
print(d2v_model.infer_vector(['I am learning NLP']))

# Prepare these vectors to be used in a machine learning model
vectors = [[d2v_model.infer_vector(words)] for words in X_test]

vectors[0]
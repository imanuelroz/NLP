import re
import nltk
import string
import pandas as pd
import numpy as np
#nltk.download()
from nltk.corpus import stopwords

desired_width = 350
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 10)


messages = pd.read_csv('Data/spam.csv', encoding='latin-1')
print(messages.head())

# Drop unused columns and label columns that will be used
messages = messages.drop(labels=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1) #axis=1 to drop columns
messages.columns = ['label', 'text'] #rename the columns
print(messages.head())
print(messages.shape)
print(messages['label'].value_counts()) #frequency per each label

messages = messages.dropna(axis=0)
print(messages.head())
print(messages.shape)

#Pre-processing Text Data
#Cleaning up the text data is necessary to highlight features that you want your
# machine learning model to use.
# 1) Remove punctuation
# 2) Tokenization
# 3) Remove stopwords

print(string.punctuation)

# 1) Remove punctuation
def remove_punct(text):
    text = ''.join([char for char in text if char not in string.punctuation])
    return text

messages['text_without_punctuation'] = messages['text'].apply(lambda x : remove_punct(x) )
print(messages.head())


#Another  way to create a new column with removed punctuation using list comprehension
# def remove_punctuation(document):
#     for i in document:
#         if i in string.punctuation:
#             text_without_punctuation = document.replace(i, '')
#             return text_without_punctuation
# messages['removed_punctuation'] = [remove_punctuation(m) for m in messages['text']]
# print(messages.head())

#2) now we tokenize using regex

def tokenize(text):
    tokens = re.split('\W+',text) #\W+ this split wherever it sees one or more non-word carachter
    return tokens

messages['tokenized_text'] = messages['text_without_punctuation'].apply(lambda x :tokenize(x.lower()))
print(messages.head())


#3) remove stopwords

stopwords = nltk.corpus.stopwords.words('english')

def remove_stopwords(tokenized_text):
    cleaned_text = [char for char in tokenized_text if char not in stopwords]
    return cleaned_text


messages['text_no_stopwords'] = messages['tokenized_text'].apply(lambda x: remove_stopwords(x))

print(messages.head())


#we can create a unique function for all the data cleaning: remove punctuation, tokenize and remove stopwords

'''
def clead(data):
    no_punct = ''.join([word.lower() for word in data if word not in string.punctuation])
    tokens = re.split('\W+', no_punct)
    no_stopwords = [word for word in tokens if word not in stopwords]

    return no_stopwords
'''



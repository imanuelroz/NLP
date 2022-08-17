import nltk
import pandas as pd
import numpy as np
#nltk.download()
#dir(nltk)
from nltk.corpus import stopwords #words which appear frequently but don't contribute to the meaning of the sentence

desired_width = 320
pd.set_option('display.width', desired_width)
#pd.set_option('display.max_colwidth', 100)
pd.set_option('display.max_columns', 10)

print(stopwords.words('english')[0:10])
print(stopwords.words('english')[0:500:25])

messages = pd.read_csv('Data/spam.csv', encoding='latin-1')
print(messages.head())

# Drop unused columns and label columns that will be used
messages = messages.drop(labels=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1) #axis=1 to drop columns
messages.columns = ['label', 'text'] #rename the columns
print(messages.head())

#check the shape:
print(messages.shape)

print(messages['label'].value_counts()) #frequency per each label

#check if there are any NULL
print('Number of NULL in label: {}'.format(messages['label'].isnull().sum()))
print('Number of NULL in text: {}'.format(messages['text'].isnull().sum()))


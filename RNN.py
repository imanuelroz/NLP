# Read in data and split into training and test set
# NOTE: we are NOT cleaning the data
import keras as keras
import numpy as np
import tensorflow as tf
import keras
import tf as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import keras.backend as K
from keras.layers import Dense, Embedding, LSTM
from keras.models import Sequential
import pandas as pd
from sklearn.model_selection import train_test_split
pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.max_columns', 10)

messages = pd.read_csv('Data/spam.csv', encoding='latin-1')
messages = messages.drop(labels=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis = 1)
messages.columns = ["label", "text"]
labels = np.where(messages['label']=='spam', 1, 0)
print(labels)

X_train, X_test, y_train, y_test = train_test_split(messages['text'],
                                                    labels, test_size=0.2)

# Initialize and fit the tokenizer  (similar purpose as the preoprocess of gensim)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

# Use that tokenizer to transform the text messages in the training and test sets
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

print(X_train_seq[0])#each integer represent a word in the text message

# Pad the sequences so each sequence is the same length

# standardize the vector since the ML model expects the same number of features for each element,
# we want vectors of same length. We do it with Pad sequences

X_train_seq_padded = pad_sequences(X_train_seq, 50) #if the sequence is longer than 50 it truncates, if it is smaller, add 0's
#the sequence represent words in a text message

X_test_seq_padded = pad_sequences(X_test_seq, 50)

# What do these padded sequences look like?
print(X_train_seq_padded[0])


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

print(len(tokenizer.index_word)+1)
#construct a simple RNN model

model = Sequential()

model.add(Embedding(len(tokenizer.index_word)+1, 32))
model.add(LSTM(32, dropout=0, recurrent_dropout=0))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()


#Compile the model

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy', precision_m, recall_m])

# Fit the RNN model
history = model.fit(X_train_seq_padded, y_train,
                    batch_size=32, epochs=10,
                    validation_data=(X_test_seq_padded, y_test))


# Plot the evaluation metrics by each epoch for the model to see if we are over or underfitting
import matplotlib.pyplot as plt

for i in ['accuracy', 'precision_m', 'recall_m']:
    acc = history.history[i]
    val_acc = history.history['val_{}'.format(i)]
    epochs = range(1, len(acc) + 1)

    plt.figure()
    plt.plot(epochs, acc, label='Training Accuracy')
    plt.plot(epochs, val_acc, label='Validation Accuracy')
    plt.title('Results for {}'.format(i))
    plt.legend()
    plt.show()
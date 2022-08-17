import gensim
import numpy as np
import pandas as pd
import gensim.downloader as api
from sklearn.model_selection import train_test_split
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.max_columns', 10)


#you can use a pretrained word2vec embedding as for example:
wiki_embeddings = api.load('glove-wiki-gigaword-100') #each vector of length 100

#print(wiki_embeddings['king'])
#print(wiki_embeddings.most_similar('king'))

# Now we train our model using the word2vec embedding,
# in a classical ML problem we think at the embedding as at the features

#First read the data:

messages = pd.read_csv('Data/spam.csv', encoding='latin-1')
messages = messages.drop(labels=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
messages.columns = ['label', 'text']
messages.head()

messages['text_clean'] = messages['text'].apply(lambda x: gensim.utils.simple_preprocess(x))
print(messages.head())

X_train, X_test, y_train, y_test = train_test_split(messages['text_clean'],
                                                    messages['label'],
                                                    test_size=0.2,
                                                    random_state=13)

#Now we train the word2vec on our data
w2v_model = gensim.models.Word2Vec(X_train, vector_size=100, window=5, min_count=2)
print(w2v_model.wv['king'])
w2v_model.wv.most_similar('king')
#How do you treat a word which is not present in the embedding?
#print(w2v_model.wv['imanuel'])
print(X_test.iloc[0]) # each row is is a list of tokens now
print(len(w2v_model.wv.index_to_key)) # all the words w2v learned a vector for

#we want to use the embedding as feature input to our ML mode
#so we generate an array of a list of arrays,
# which contains lists corresponding to each word arr([arr([first_word_first_doc][sec_word_first_doc]...),
#                                                      arr([first_word_second_doc][sec_word_second_doc]...),
#                                                      ....])

w2v_vect = np.array([np.array([w2v_model.wv[i] for i in ls if i in w2v_model.wv.index_to_key])for ls in X_test],
                    dtype=object)

for i, v in enumerate(w2v_vect):
    print(len(X_test.iloc[i]), len(v))  #len(v) = how many word vector do we have for each document


w2v_vect_avg = []

for vect in w2v_vect:
    if len(vect)!=0:
        w2v_vect_avg.append(vect.mean(axis=0))
    else:
        w2v_vect_avg.append((np.zeros(100)))

for i, v in enumerate(w2v_vect_avg):
    print(len(X_test.iloc[i]), len(v)) #len(v) = how many word vector do we have 1 for each document of length 100
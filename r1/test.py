import numpy as np
from pickle import load
from keras.preprocessing import sequence, text
from keras.models import Sequential
from keras.layers import Dense, Embedding, SpatialDropout1D, LSTM, Dropout
from keras.regularizers import l2
import re

max_headline_length = 70
word_count = 20740

model = Sequential()
model.add(Embedding(word_count, 32, input_length=max_headline_length))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(64, kernel_regularizer=l2(0.005), dropout=0.3, recurrent_dropout=0.3))
model.add(Dropout(0.5))
model.add(Dense(32, kernel_regularizer=l2(0.005)))
model.add(Dropout(0.5))
model.add(Dense(2, kernel_regularizer=l2(0.001), activation='softmax'))

model.load_weights('model.h5')
word_to_index = load(open('words.pkl', 'rb'))

def get_words(string):
  words = []
  for word in re.finditer("[a-z]+|[\"'.;/!?]", string.lower()):
    words.append(word.group(0))
  return words

def words_to_indexes(words):
  return [word_to_index.get(word, 0) for word in words]

def format_input(word_indexes):
  return sequence.pad_sequences([word_indexes], maxlen=max_headline_length)[0]

def get_type(string):
  words = words_to_indexes(get_words(string))
  result = model.predict(np.array([format_input(words)]))[0]

  if result[0] > result[1]:
    site = 'NotTheOnion'
  else:
    site = 'TheOnion'

  return site

import sys
print(get_type(sys.argv[1]))

# -*- coding: utf-8 -*-
"""Analysis.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ds9i9R1cXmvPXxwfRQO6hrLyM500CD1S
"""

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
import re
from gensim.models import Word2Vec
from tensorflow import keras
nltk.download('stopwords')
nltk.download('punkt')

xbox = pd.read_csv('Tweets-xbox.csv', encoding='latin-1')
ps = pd.read_csv('Tweets-ps.csv', encoding='latin-1')

POS_WORDS_PATH = 'positive_words.txt'
NEG_WORDS_PATH = 'negative_words.txt'
WINDOW_SIZE = 100

def parse_word_file(filename):
  words = set()
  with open(filename) as f:
    for line in f:
        words.add(line.strip())
  return words


POS_WORDS = parse_word_file(POS_WORDS_PATH)
NEG_WORDS = parse_word_file(NEG_WORDS_PATH)

xbox_text = xbox['text']
ps_text = ps['text']

stop_words = list(stopwords.words('english'))

# for user mentions
at_pattern = '@\S+'
# for non_alphanumberic charachters
non_alpha = '[^A-Za-z0-9]+'
# urls
url = 'https?:\S+|http?:\S+'


def prepare(text):
  # strip out the mentions
  text = re.sub(at_pattern, ' ', text.lower())
  text = re.sub(non_alpha, ' ', text)
  text = re.sub(url, ' ', text)
  text = text.strip()
  tokens = []
  for word in text.split():
    if word not in stop_words:
      tokens.append(word)
  return " ".join(tokens)


xbox_text = xbox_text.apply(lambda text : prepare(text))
ps_text = ps_text.apply(lambda text : prepare(text))

DATA_PATH = 'Tweets-all.csv'

df = pd.read_csv(DATA_PATH, encoding='latin-1')

# construct the corpus treating each individual tweet as a document.
corpus = [word_tokenize(tweet) for tweet in df['text']]
print(len(corpus))
# figure out what we want for the w2vec params
w2v = Word2Vec(corpus, size=WINDOW_SIZE, window=5,  min_count=5, workers=4)
# train the model
w2v.train(corpus, total_examples=len(w2v.wv.vocab), epochs=10)


def count_word_sentiment(text, words):
  count = 0
  for word in text.split():
    if word in words:
      count += 1
  return count


def encode_tweet(model, tweet):
  tokens = word_tokenize(tweet)
  word_counts = np.array([count_word_sentiment(tweet, POS_WORDS), count_word_sentiment(tweet, NEG_WORDS)])
  vec = np.zeros(WINDOW_SIZE).reshape(1, WINDOW_SIZE)
  for w in tokens:
    try:
      vec = np.add(vec, model[w].reshape(1, WINDOW_SIZE))
    except KeyError:
      # ignore for now
      continue
  vec /= len(tokens)
  vec = vec.flatten()
  vec = np.concatenate((vec, word_counts))
  return pd.Series(vec)

# Store tweets to analyze
X_xbox = xbox_text.apply(lambda tweet : encode_tweet(w2v, tweet))
X_ps = ps_text.apply(lambda tweet : encode_tweet(w2v, tweet))

#Load neural net model
model = keras.models.load_model('dropoutsequentialmodel')

# Get sentiment value for tweets
Y_xbox = model.predict(X_xbox)
Y_ps = model.predict(X_ps)

result_xbox = pd.concat([pd.DataFrame(Y_xbox), xbox['created_at']], axis=1, sort=False)
result_ps = pd.concat([pd.DataFrame(Y_ps), ps['created_at']], axis=1, sort=False)

def get_date(text):
  split = text.split(' ')
  return split[0]

# Store only date part of tweet
result_ps['date'] = result_ps['created_at'].apply(lambda text : get_date(text))
result_xbox['date'] = result_xbox['created_at'].apply(lambda text : get_date(text))

datewise_xbox = result_xbox.groupby(result_xbox['date']).mean()
datewise_ps = result_ps.groupby(result_ps['date']).mean()

#Combine the two datasets
final = pd.concat([datewise_xbox, datewise_ps[0]], axis=1, sort=False)
final.columns = ['Xbox', 'PS5']
final.index = pd.to_datetime(final.index).strftime('%d-%m-%Y')
final.to_csv('final.csv',index = True)

# Divide into positive and negative sentiment
result_ps['sentiment'] = result_ps[0].apply(lambda x: True if x>=0.5 else False)
result_xbox['sentiment'] = result_xbox[0].apply(lambda x: True if x>=0.5 else False)

# Count the positive and negative sentiment for each date
sentiment_ps = result_ps.groupby(['date', 'sentiment']).size().unstack(fill_value=0)
sentiment_xbox = result_xbox.groupby(['date', 'sentiment']).size().unstack(fill_value=0)

# Get ratio of negative to positive sentiment
sentiment_ps['PS5'] = sentiment_ps[sentiment_ps.columns[0]]/sentiment_ps[sentiment_ps.columns[1]]
sentiment_xbox['Xbox'] = sentiment_xbox[sentiment_xbox.columns[0]]/sentiment_xbox[sentiment_xbox.columns[1]]
# Combine results
comparison = pd.concat([sentiment_xbox, sentiment_ps[sentiment_ps.columns[2]]], axis=1, sort=False)

del comparison[comparison.columns[0]]
del comparison[comparison.columns[0]]

comparison.to_csv('comparison.csv',index = True)

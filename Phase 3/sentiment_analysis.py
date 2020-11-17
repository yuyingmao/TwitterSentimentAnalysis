"""
Sentiment Analysis.py
Data Preparation for Sentiment 140 dataset
with augmented word2vecs for sentiment counts
"""

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
import re
import string
from sklearn.model_selection import train_test_split
import gensim
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.models import Sequential
from keras.layers import Dense

nltk.download('stopwords')
nltk.download('punkt')

# TODO: Update these paths appropriately
DATA_PATH = '/gdrive/My Drive/sentiment_data/training.csv'
POS_WORDS_PATH = '/gdrive/My Drive/sentiment_data/positive_words.txt'
NEG_WORDS_PATH = '/gdrive/My Drive/sentiment_data/negative_words.txt'
WINDOW_SIZE = 100
# the columns for the data
COLUMNS = ['target', 'id', 'date', 'flag', 'user', 'text']


def parse_word_file(filename):
    words = set()
    with open(filename) as f:
        for line in f:
            words.add(line.strip())
    return words


POS_WORDS = parse_word_file(POS_WORDS_PATH)
NEG_WORDS = parse_word_file(NEG_WORDS_PATH)
# print(len(POS_WORDS))
# print(len(NEG_WORDS))
# print(POS_WORDS)
# print(NEG_WORDS)

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


def load_dataset():
    df = pd.read_csv(DATA_PATH, encoding='latin-1', names=COLUMNS)
    # add the correct columns to the dataframe
    df.columns = ['target', 'id', 'date', 'flag', 'user', 'text']
    # we only care about the prediction value and the text of the tweet
    df = df.drop(columns=['id', 'date', 'flag', 'user'])
    # normalize the space between 0-1
    df['target'] = df['target'].replace(4, 1)
    # df.head()
    df.text = df.text.apply(lambda text: prepare(text))
    return df


def train_w2v(df):
    """
    Train a word2vec model with the preprocessed data
    :param df: the dataframe containing the text of the dataset
    :return: the trained word2vec model
    """
    # construct the corpus treating each individual tweet as a document
    corpus = [word_tokenize(tweet) for tweet in df['text']]
    print(len(corpus))
    w2v = Word2Vec(corpus, size=WINDOW_SIZE, window=5, min_count=5, workers=4)
    # train the model
    w2v.train(corpus, total_examples=len(w2v.wv.vocab), epochs=10)
    # TODO update this path to save on local machine
    w2v.wv.save('/gdrive/My Drive/sentiment_data/word2vecmodel')
    return w2v


def count_word_sentiment(text, words):
    """
    Counts the number of words occurring with a sentiment in the set provided
    :param text: the tweet to analyze for counts
    :param words: the words to search for
    :return: the number of sentiment specific words in the text
    """
    count = 0
    for word in text.split():
        if word in words:
            count += 1
    return count


def encode_tweet(model, tweet):
    """
    Vectorizes a tweet with word2vec and augments it with sentiment counts
    :param model: the word2vec model
    :param tweet: the tweet to encode
    :return: the vectorized tweet
    """
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


def split_and_save(df, w2v):
  """
  encodes the tweets into vectors, splits into training and test split and saves them for training
  :param df: the preprocessed tweets
  :param w2v: the word2vec model
  """
  X = df.text.apply(lambda tweet: encode_tweet(w2v, tweet))
  y = df['target']
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=100)
  print("Training Split: ")
  print("X: " + str(X_train.shape))
  print("Y: " + str(y_train.shape))

  print("Test Split: ")
  print("X: " + str(X_test.shape))
  print("Y: " + str(y_test.shape))

  print(y_train.shape)
  print(y_test.shape)
  #y_train.to_csv('/gdrive/My Drive/sentiment_data/train_targets.csv')
  #y_test.to_csv('/gdrive/My Drive/sentiment_data/test_targets.csv')
  # TODO: update this path appropriately for a local machine
  np.save('/gdrive/My Drive/sentiment_data/train_targets.npy', y_train.to_numpy())
  np.save('/gdrive/My Drive/sentiment_data/test_targets.npy', y_test.to_numpy())

  print(X_train.shape)
  print(X_test.shape)
  # TODO: update this path appropriately for a local machine
  X_train.to_csv('/gdrive/My Drive/sentiment_data/encoded_train.csv', index=False)
  X_test.to_csv('/gdrive/My Drive/sentiment_data/encoded_test.csv', index=False)

def main():
  df = load_dataset()
  w2v = train_w2v(df)
  split_and_save(df, w2v)


if __name__ == '__main__':
    main()
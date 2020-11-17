"""
Training.py
loads input data and tests various learning methods
and reports their accuracy on training and test data
"""

import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools


def load_data():
    """
    Loads training and test data from provided paths
    :return: the training and tests data and labels
    """
    # TODO: update with path to encoded train and test csv and targets
    X_train = pd.read_csv('/gdrive/My Drive/sentiment_data/encoded_train.csv')
    X_test = pd.read_csv('/gdrive/My Drive/sentiment_data/encoded_test.csv')
    y_train = np.load('/gdrive/My Drive/sentiment_data/train_targets.npy')
    y_test = np.load('/gdrive/My Drive/sentiment_data/test_targets.npy')
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)
    return X_train, X_test, y_train, y_test


def naive_bayes(X_train, X_test, y_train, y_test):
    """
    Tests Naive Bayes for accuracy on train and test
    :param X_train: the training data
    :param X_test:  the testing data
    :param y_train: the training labels
    :param y_test:  the testing labels
    """
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    print("accuracy on train: " + str(nb.score(X_train, y_train)))
    print("accuracy on test: " + str(nb.score(X_test, y_test)))
    return nb


def decicion_tree(X_train, X_test, y_train, y_test):
    """
    Tests decision tree classifiers for accuracy on train and test
    :param X_train: the training data
    :param X_test:  the testing data
    :param y_train: the training labels
    :param y_test:  the testing labels
    """
    dt = tree.DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    # tree.plot_tree(DT)
    print("accuracy on train: " + str(dt.score(X_train, y_train)))
    print("accuracy on test: " + str(dt.score(X_test, y_test)))


def random_forest(X_train, X_test, y_train, y_test):
    """
    Tests decision tree classifiers for accuracy on train and test
    :param X_train: the training data
    :param X_test:  the testing data
    :param y_train: the training labels
    :param y_test:  the testing labels
    """
    rfc = RandomForestClassifier(max_depth=3, random_state=0)
    rfc.fit(X_train, y_train)
    print("accuracy on train: " + str(rfc.score(X_train, y_train)))
    print("accuracy on test: " + str(rfc.score(X_test, y_test)))


def neural_net(X_train, X_test, y_train, y_test):
    model = Sequential()
    model.add(Dense(100, activation='relu', input_dim=102))
    model.add(Dropout(.3))
    model.add(Dense(100))
    model.add(Dropout(.5))
    model.add(Dense(1, activation='sigmoid'))

    model.summary()

    ES = EarlyStopping(monitor='loss', patience=3)
    # TODO: update path
    CP = ModelCheckpoint(
        filepath='/gdrive/My Drive/sentiment_data/savedmodel',
        save_weights_only=True,
        monitor='val_acc',
        mode='max',
        save_best_only=True)

    model.compile(loss='binary_crossentropy',
                  optimizer="adam",
                  metrics=['accuracy'])
    plot_model(model)

    history = model.fit(X_train, y_train,
                        batch_size=100,
                        epochs=10,
                        verbose=1, callbacks=[ES, CP], validation_data=(X_test, y_test))

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    labels = model.predict(X_test)
    labels[labels > .5] = 1
    labels[labels <= .5] = 0

    cm = confusion_matrix(y_true=y_test, y_pred=labels)
    plot_confusion_matrix(cm, classes=['negative', 'positive'], title="Confusion Matrix for Model")
    # TODO: update with local path to save model
    model.save('/gdrive/My Drive/sentiment_data/dropoutsequentialmodel')


# from SICKIT LEARN -> plot confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def main():
    # run the training
    X_train, X_test, y_train, y_test = load_data()
    naive_bayes(X_train, X_test, y_train, y_test)
    decicion_tree(X_train, X_test, y_train, y_test)
    random_forest(X_train, X_test, y_train, y_test)
    neural_net(X_train, X_test, y_train, y_test)

if __name__ == '__main__':
    main()
"""
Spam Messages detection using LSTM and BiLSTM with data augmentation
"""

import logging
import string
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, LSTM, Bidirectional, Embedding
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.preprocessing import sequence
from textattack.augmentation import EmbeddingAugmenter

logging.basicConfig(level=logging.INFO)

# This variable is set True if we want to use the already augmented data which
# is stored in the outputs dir. If we want to perform the augmentation again
# then set this variable False and perform_aug=True at the end of file
# NOTE: If the augmentation is performed again, then the results of the models
# can be slightly different because the Augmenter generates different data
# each time
AUGMENTED_DATA = False


def remove_stopwords(text):
    """
    Remove stop words from the text and words having length < 3
    :param text: string
    :return: string
    """
    try:
        # load stop words from nltk corpus
        stop_words = stopwords.words("english")
        # list of words
        text = text.split()
        # filter stop words and words with length < 3
        text = [word for word in text if len(word) > 2
                and word not in stop_words]
        # to string
        text = " ".join(text)
        return text
    except Exception as e:
        logging.error(e)


def remove_punctuation(text):
    """
    Removes punctuation from the string
    :param text: string
    :return: string
    """
    try:
        # Below code line is used from the following link
        # https://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string#:~:text=s.translate(str.maketrans(%27%27%2C%20%27%27%2C%20string.punctuation))
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text
    except Exception as e:
        logging.error(e)


def remove_digits(text):
    """
    Removes digits from the string
    :param text: string
    :return: string
    """
    try:
        # filter digits for each character
        # code is used from the following link
        # https://stackoverflow.com/questions/12851791/removing-numbers-from-string#:~:text=result%20%3D%20%27%27.join(%5Bi%20for%20i%20in%20s%20if%20not%20i.isdigit()%5D)
        text = [char for char in text if not char.isdigit()]
        # to string
        text = "".join(text)
        return text
    except Exception as e:
        logging.error(e)


def display_common_words(data, k):
    """
    Display the k most common words for both spam and ham messages
    :param data: dataframe
    :param k: k common words
    :return: None
    """
    try:
        # get ham messages
        ham_data = data[data['v1'] == 'ham']
        # get spam messages
        spam_data = data[data['v1'] == 'spam']

        # get most k common words
        # Below two lines are used from the following link
        # https://stackoverflow.com/questions/29903025/count-most-frequent-100-words-from-sentences-in-dataframe-pandas#:~:text=Counter(%22%20%22.join(df%5B%22text%22%5D).split()).most_common(100)
        ham_words = Counter(" ".join(ham_data["v2"]).split()).most_common(k)
        spam_words = Counter(" ".join(spam_data["v2"]).split()).most_common(k)

        # convert list of words and counts to dataframe
        ham_df = pd.DataFrame(ham_words)
        spam_df = pd.DataFrame(spam_words)

        # save ham words plot
        plt.title("{} most common words in ham messages".format(k))
        sns.barplot(y=0, data=ham_df, x=1)
        plt.savefig("figs/ham_words.jpg")

        # save spam words plot
        plt.title("{} most common words in spam messages".format(k))
        sns.barplot(y=0, data=spam_df, x=1)
        plt.savefig("figs/spam_words.jpg")

    except Exception as e:
        logging.error(e)


def clean_text(text):
    """
    Clean text by removing punctuation, digits, and stop words
    :param text: string
    :return: string
    """
    try:
        # to lower case
        text = text.lower()
        # remove punctuation
        text = remove_punctuation(text)
        # remove digits
        text = remove_digits(text)
        # remove stop words
        text = remove_stopwords(text)

        return text
    except Exception as e:
        logging.error(e)


def text_aug(texts):
    """
    Perform augmentation
    :param texts: list of texts
    :return: list of augmented texts
    """
    try:
        # to store augmentations
        augmentations = []
        # initialize augmenter to generate 6 new samples for each text
        aug = EmbeddingAugmenter(transformations_per_example=6)
        # augment and append data for each text
        for text in texts:
            augmentations = augmentations + aug.augment(text)

        return augmentations
    except Exception as e:
        logging.error(e)


def balance_data(data):
    """
    Balance data by using the augmentation
    :param data: dataframe
    :return: dataframe
    """
    try:
        # get spam messages
        spam = data[data['v1'] == 'spam']
        # list of spam messages
        texts = spam['v2'].tolist()

        # call augmentation method
        print("Augmenting data... It will take a few minutes.")
        aug = text_aug(texts)

        # create dataframe for augmented data
        aug_data = pd.DataFrame()
        aug_data['v2'] = aug
        aug_data['v1'] = ['spam'] * len(aug)

        # concat augmented data with original data
        data = pd.concat([aug_data, data])
        # store combined data
        data.to_csv("outputs/augmented_data.csv", index=False)
        return data
    except Exception as e:
        logging.error(e)


def get_sequences(data):
    """
    Tokenize and prepare data for the models
    :param data: texts
    :return: text sequences and vocabulary size
    """
    try:
        # As I never worked on LSTM models before, I referred to the
        # following link to see the use of Tokenizer which is standard
        # for LSTMs
        # https://www.kaggle.com/code/kredy10/simple-lstm-for-text-classification?scriptVersionId=2648057&cellId=14
        # initialize tokenizer
        tokenizer = Tokenizer()
        # fit tokenizer on data
        tokenizer.fit_on_texts(data)
        # generate sequences
        sequences = tokenizer.texts_to_sequences(data)
        # pad sequences for same length
        padded_seq = sequence.pad_sequences(sequences)
        # get vocab size for embedding layer as mentioned in the documentation
        # https://keras.io/api/layers/core_layers/embedding/#:~:text=input_dim%3A%20Integer.%20Size%20of%20the%20vocabulary%2C%20i.e.%20maximum%20integer%20index%20%2B%201
        vocabulary_size = len(tokenizer.word_index) + 1

        return padded_seq, vocabulary_size

    except Exception as e:
        logging.error(e)


def get_callbacks():
    """
    Callbacks for deep learning models
    :return: callbacks list
    """
    try:
        # To monitor validation loss if it does not decrease for two
        # epochs with 0.0001 minimum change
        loss_callback = EarlyStopping(monitor='val_loss', patience=2,
                                      mode='min', min_delta=0.0001)
        # To monitor validation accuracy if it does not increase for three
        # epochs with 0.001 minimum change
        acc_callback = EarlyStopping(monitor='val_accuracy', patience=3,
                                     mode='max', min_delta=0.001)
        # list of callbacks
        callbacks = [loss_callback, acc_callback]

        return callbacks

    except Exception as e:
        logging.error(e)


def write_scores(predictions, score, models, aug):
    """
    Create dataframe to validate error rates
    """
    try:
        # calculate average of prediction scores
        pred_arr = np.array(predictions)
        mean = np.average(pred_arr, axis=0)

        # create dataframe for models and folds
        df = pd.DataFrame(pred_arr.transpose(),
                          columns=['fold1', 'fold2', 'fold3',
                                   'fold4', 'fold5'])
        df['models'] = models
        df.set_index('models', inplace=True)
        df['mean'] = mean

        # If already augmented data is used or new augmentation is performed
        if AUGMENTED_DATA or aug:
            file_names = ["accuracy_aug.json", "f1_aug.json",
                          "roc_auc_aug.json"]
        else:
            # for original dataset
            file_names = ["accuracy.json", "f1.json", "roc_auc.json"]

        # write scores for each metric
        if score == 'acc':
            df.to_json("outputs/{}".format(file_names[0]), indent=4)
        if score == "f1":
            df.to_json("outputs/{}".format(file_names[1]), indent=4)
        if score == "roc_auc":
            df.to_json("outputs/{}".format(file_names[2]), indent=4)

    except Exception as e:
        logging.error(e)


def lstm(voc_size):
    """
    LSTM model implementation
    Initial idea for LSTM was taken form below link but
    several changes are made based on different experiments
    https://www.geeksforgeeks.org/lstm-based-poetry-generation-using-nlp-in-python/#:~:text=%23%20Building%20a%20Bi,(model.summary())
    :param voc_size: vocabulary size for embedding layer
    :return: LSTM model
    """
    try:
        # instantiate sequential model
        model = Sequential()
        # embedding layer with input_dim=vocabulary_size and output_dim=100
        model.add(Embedding(voc_size, 100))
        # LSTM with 128 units and returning sequences
        model.add(LSTM(128, return_sequences=True))
        # dropout to avoid overfitting
        model.add(Dropout(0.35))
        # LSTM with 32 units and returning sequences
        model.add(LSTM(32, return_sequences=True))
        # dropout to avoid overfitting
        model.add(Dropout(0.35))
        # LSTM with 16 units
        model.add(LSTM(16, return_sequences=False))
        # fully connected layers with a dropout layer in between
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation="sigmoid"))
        # Adam optimizer is used
        model.compile(loss="binary_crossentropy",
                      optimizer='adam', metrics=["accuracy"])
        return model
    except Exception as e:
        logging.error(e)


def bi_lstm(voc_size):
    """
    BiLSTM with similar architecture to LSTM defined above but different units
    :param voc_size: vocabulary size for embedding layer
    :return: BiLSTM model
    """
    try:
        # instantiate sequential model
        model = Sequential()
        # embedding layer with input_dim=vocabulary_size and output_dim=100
        model.add(Embedding(voc_size, 100))
        # BiLSTM with 64 units and returning sequences
        model.add(Bidirectional(LSTM(64, return_sequences=True)))
        # dropout to avoid overfitting
        model.add(Dropout(0.5))
        # BiLSTM with 32 units and returning sequences
        model.add(Bidirectional(LSTM(32, return_sequences=True)))
        # dropout to avoid overfitting
        model.add(Dropout(0.5))
        # BiLSTM with 16 units
        model.add(Bidirectional(LSTM(16, return_sequences=False)))
        # fully connected layers with a dropout layer in between
        model.add(Dense(64, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation="sigmoid"))
        # Adam optimizer is used
        model.compile(loss="binary_crossentropy",
                      optimizer='adam', metrics=["accuracy"])
        return model
    except Exception as e:
        logging.error(e)


def model_prediction(models, train_x, train_y, test_x, test_y):
    """
    Train models and get prediction scores
    :param models: list of instantiated models
    :param train_x: training texts
    :param train_y: training labels
    :param test_x: testing texts
    :param test_y: testing labels
    :return: lists of scores
    """
    try:
        # lists for scores
        acc_scores = []
        f1_scores = []
        roc_auc_scores = []

        # get callbacks
        callbacks = get_callbacks()
        # train and test each model
        for model in models:
            model.fit(train_x, train_y, epochs=10,
                      validation_split=0.15, callbacks=callbacks)
            # used the below code line to get model predictions from following link
            # https://stackoverflow.com/questions/68836551/keras-attributeerror-sequential-object-has-no-attribute-predict-classes#:~:text=last%2Dlayer%20activation).*-,(model.predict(x)%20%3E%200.5).astype(%22int32%22),-%2C%20if%20your%20model
            pred = (model.predict(test_x) > 0.5).astype("int32")

            # accuracy, f1, roc auc metrics
            acc = accuracy_score(pred, test_y)
            f1 = f1_score(pred, test_y)
            roc_auc = roc_auc_score(pred, test_y)

            # append scores
            acc_scores.append(acc)
            f1_scores.append(f1)
            roc_auc_scores.append(roc_auc)

        return acc_scores, f1_scores, roc_auc_scores
    except Exception as e:
        logging.error(e)


def spam_detection(data, perform_aug=False, plots=False):
    try:
        # discard useless columns
        data = data[['v1', 'v2']]

        # check for augmentation
        if AUGMENTED_DATA is False:
            if perform_aug:
                data = balance_data(data)

        # clean text messages
        data['v2'] = data['v2'].apply(clean_text)

        # 20 most common words for spam and ham
        if plots:
            display_common_words(data, 20)

        # label encoder for spam and ham
        le = preprocessing.LabelEncoder()
        target = le.fit_transform(data['v1'])

        # get text sequences and vocab size
        text = np.array(data['v2'])
        sequences, voc_size = get_sequences(text)
        target = np.array(target)

        # cross validation technique
        skf = StratifiedKFold(n_splits=5, random_state=None,
                              shuffle=False)

        # to store scores for each fold
        acc_scores = []
        f1_scores = []
        roc_auc_scores = []
        for _, (train_index, test_index) in enumerate(
                skf.split(sequences, target)):
            # extract train and test data based on indices
            train_x = sequences[train_index]
            train_y = target[train_index]
            test_x = sequences[test_index]
            test_y = target[test_index]

            # instantiate LSTM and BiLSTM
            lstm_model = lstm(voc_size)
            bi_lstm_model = bi_lstm(voc_size)
            models = [lstm_model, bi_lstm_model]

            # train and predict
            acc, f1, roc_auc = model_prediction(models, train_x,
                                                train_y, test_x, test_y)
            # append scores for each metric
            acc_scores.append(acc)
            f1_scores.append(f1)
            roc_auc_scores.append(roc_auc)

        # write scores for each metric in outputs dir
        models = ["lstm", "bi_lstm"]
        write_scores(acc_scores, "acc", models, perform_aug)
        write_scores(f1_scores, "f1", models, perform_aug)
        write_scores(roc_auc_scores, "roc_auc", models, perform_aug)

    except Exception as e:
        logging.error(e)


if __name__ == '__main__':

    # If you want to use the already augmented data and do not
    # want to perform augmentation again
    if AUGMENTED_DATA:
        sms_data = pd.read_csv("outputs/augmented_data.csv")
    else:
        # original dataset
        sms_data = pd.read_csv("data/spam.csv")

    # Perform spam detection
    # perform_aug is used to run augmenter again if set True and AUGMENTED_DATA
    # is False
    # generate plots for word counts if plots=True
    spam_detection(sms_data, perform_aug=False, plots=False)

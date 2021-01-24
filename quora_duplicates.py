import csv
import json
from os.path import exists, expanduser
from zipfile import ZipFile

import pandas
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras import backend
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.backend import concatenate
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.layers import TimeDistributed, Lambda, Dense, Dropout, BatchNormalization
from tensorflow.python.keras.layers.embeddings import Embedding
from tensorflow.python.keras.utils.data_utils import get_file
import numpy as np


def separate_data(dataset, pred_data):
    print('Separating Data')
    dataset.dropna(inplace=True)
    q1 = dataset.Question1.tolist()
    q2 = dataset.Question2.tolist()
    is_dup = dataset.IsDuplicate.tolist()

    # pred_data.dropna(inplace=True)
    q1_pred = pred_data.Question1.tolist()
    q2_pred = pred_data.Question2.tolist()

    return q1, q2, is_dup, q1_pred, q2_pred


def read_data(rows):
    print('Reading Data')
    train_data = pandas.read_csv("datasets2020/datasets/q2b/train.csv", delimiter=',', nrows=rows)
    test_data = pandas.read_csv("datasets2020/datasets/q2b/test_without_labels.csv", delimiter=',', nrows=rows)
    return train_data, test_data


def tokenize(q1, q2, max_tokenized_words):
    print('Tokenizing Data')
    questions = q1 + q2

    tokenizer = Tokenizer(num_words=max_tokenized_words)

    # Update Vocabulary with each word
    tokenizer.fit_on_texts(questions)

    # Convert questions to a sequence of integers (word id's)
    question1_word_sequences = tokenizer.texts_to_sequences(q1)
    question2_word_sequences = tokenizer.texts_to_sequences(q2)

    # Make an ordered dictionary of sorted words (high word_count first) {word: id}
    word_index = tokenizer.word_index

    return question1_word_sequences, question2_word_sequences, word_index


def process_glove():
    print('Processing GloVe data')
    if not exists(expanduser('~/.keras/datasets/') + 'glove.840B.300d.zip'):
        zipfile = ZipFile(get_file('glove.840B.300d.zip', 'http://nlp.stanford.edu/data/glove.840B.300d.zip'))
        zipfile.extract('glove.840B.300d.txt', path='~/.keras/datasets/')

    embeddings_index = {}
    with open('~/.keras/datasets/glove.840B.300d.txt', encoding='utf-8') as f:
        for line in f:
            values = line.split(' ')
            word = values[0]
            embedding = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = embedding

    return embeddings_index


def zero_pad(q1, q2, is_dup, max_sentence_words):
    print('Adding Padding to Data')
    q1_data = pad_sequences(q1, maxlen=max_sentence_words)
    q2_data = pad_sequences(q2, maxlen=max_sentence_words)
    labels = None
    if is_dup:
        labels = np.array(is_dup, dtype=int)
        return q1_data, q2_data, labels
    return q1_data, q2_data


def embedding_matrix(word_index, embeddings_index, embedding_dim, max_tokenized_words):
    print('Creating Embedding matrix')
    nb_words = min(max_tokenized_words, len(word_index))
    word_embedding_matrix = np.zeros((nb_words + 1, embedding_dim))
    for word, i in word_index.items():
        if i > max_tokenized_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            word_embedding_matrix[i] = embedding_vector

    return word_embedding_matrix, nb_words


def preprocess_data(q1, q2, is_duplicate, q1_pred, q2_pred, max_tokenized_words, embedding_dim, max_sentence_words):
    # Tokenize sentences into words
    question1_word_sequences, question2_word_sequences, word_index = tokenize(q1, q2, max_tokenized_words)
    question1_pred_word_sequences, question2_pred_word_sequences, pred_word_index = tokenize(q1_pred, q2_pred, max_tokenized_words)

    # Create an embeddings dictionary from Glove embeddings
    embeddings_index = process_glove()

    # Create word embedding matrix according to Glove embeddings and our dataset
    word_embedding_matrix, nb_words = embedding_matrix(word_index, embeddings_index, embedding_dim, max_tokenized_words)

    # Insert zero paddings
    q1_data, q2_data, labels = zero_pad(question1_word_sequences, question2_word_sequences, is_duplicate, max_sentence_words)
    q1_pred_data, q2_pred_data = zero_pad(question1_pred_word_sequences, question2_pred_word_sequences, None, max_sentence_words)

    return q1_data, q2_data, labels, q1_pred_data, q2_pred_data, word_embedding_matrix, nb_words


def split_train_set(q1_data, q2_data, labels):
    q_data = np.stack((q1_data, q2_data), axis=1)
    x_train, x_test, y_train, y_test = train_test_split(q_data, labels, test_size=0.1, random_state=55964338)
    Q1_train = x_train[:, 0]
    Q2_train = x_train[:, 1]
    Q1_test = x_test[:, 0]
    Q2_test = x_test[:, 1]

    return Q1_train, Q2_train, Q1_test, Q2_test, y_train, y_test


def apply_embeddings(word_embedding_matrix, nb_words, embedding_dim, max_sentence_words):
    # Create the Tensor
    question = Input(shape=(max_sentence_words,))

    # Embedding layer used with the GloVe predefined weights in order to load the pretrained data
    q = Embedding(nb_words + 1, embedding_dim, weights=[word_embedding_matrix], input_length=max_sentence_words, trainable=False)(question)

    # In every word of the sentence (sequence) apply the same dense layer
    q = TimeDistributed(Dense(embedding_dim, activation='relu'))(q)

    # Add a layer that outputs the max input
    q = Lambda(lambda x: backend.max(x, axis=1), output_shape=(embedding_dim,))(q)

    return question, q


def make_model(word_embedding_matrix, nb_words, activation, loss_func, embedding_dim, dropout, optimizer, layers, max_sentence_words):
    print('Creating the model')

    # Create Tensors and create Embedding and Time Distributed Dense layer
    question1, q1 = apply_embeddings(word_embedding_matrix, nb_words, embedding_dim, max_sentence_words)
    question2, q2 = apply_embeddings(word_embedding_matrix, nb_words, embedding_dim, max_sentence_words)

    # Merge the outputs of the layers of the two questions
    merged = concatenate([q1, q2])

    # Create x dense layers followed by dropout and batch normalization layers
    for _ in range(layers):
        merged = Dense(200, activation=activation)(merged)

        # Randomly set input units to 0 with dropout frequency
        merged = Dropout(dropout)(merged)

        # mean output close to 0 and std close to 1
        merged = BatchNormalization()(merged)

    # Create the output layer
    is_duplicate = Dense(1, activation='sigmoid')(merged)

    model = Model(inputs=[question1, question2], outputs=is_duplicate)
    model.compile(loss=loss_func, optimizer=optimizer, metrics=['accuracy'])

    return model


def evaluate(model, Q1_test, Q2_test, y_test):
    model.load_weights('question_pairs_weights.h5')
    loss, accuracy = model.evaluate([Q1_test, Q2_test], y_test, verbose=0)
    print('Test loss = {0:.4f}, test accuracy = {1:.4f}'.format(loss, accuracy))


def save_data(q1_data, q2_data, labels, q1_pred_data, q2_pred_data, word_embedding_matrix, nb_words):
    np.save(open('q1_train.npy', 'wb'), q1_data)
    np.save(open('q2_train.npy', 'wb'), q2_data)
    np.save(open('label_train.npy', 'wb'), labels)
    np.save(open('q1_pred_data.npy', 'wb'), q1_pred_data)
    np.save(open('q2_pred_data.npy', 'wb'), q2_pred_data)
    np.save(open('word_embedding_matrix.npy', 'wb'), word_embedding_matrix)
    with open('nb_words.json', 'w') as f:
        json.dump({'nb_words': nb_words}, f)


def load_data():
    q1_data = np.load(open('q1_train.npy', 'rb'))
    q2_data = np.load(open('q2_train.npy', 'rb'))
    labels = np.load(open('label_train.npy', 'rb'))
    q1_pred_data = np.load(open('q1_pred_data.npy', 'rb'))
    q2_pred_data = np.load(open('q2_pred_data.npy', 'rb'))
    word_embedding_matrix = np.load(open('word_embedding_matrix.npy', 'rb'))
    with open('nb_words.json', 'r') as f:
        nb_words = json.load(f)['nb_words']

    return q1_data, q2_data, labels, q1_pred_data, q2_pred_data, word_embedding_matrix, nb_words


def predict(model, q1_pred_data, q2_pred_data):
    # Load trained weights
    model.load_weights('question_pairs_weights.h5')

    # Make predictions
    predictions = model.predict([q1_pred_data, q2_pred_data])

    # Generate classes from predictions
    duplicates = list()
    for pred in predictions:
        dup = 0
        if pred[0] > 0.5:
            dup = 1
        duplicates.append(dup)

    return duplicates


def save_predictions(predictions):
    ids = [i for i in range(283003, 283003+len(predictions))]

    with open('results_duplicates.csv', 'w') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(("Id", "Predicted"))
        wr.writerows(zip(ids, predictions))


if __name__ == '__main__':
    # Parameters
    rows = None
    embedding_dim = 300
    max_tokenized_words = 200000
    dropout = 0.1
    optimizer = 'adam'
    layers = 4
    activation = 'relu'
    loss_func = 'binary_crossentropy'
    epochs = 30
    batch_size = 64
    max_sentence_words = 40
    data_processed = 1
    model_trained = 0

    if data_processed:
        # Load preprocessed data from saved files
        q1_data, q2_data, labels, q1_pred_data, q2_pred_data, word_embedding_matrix, nb_words = load_data()

    else:
        # Read Dataset
        train_data, pred_data = read_data(rows)

        # Separate Train set into Question1 column, Question2 column, Labels column
        q1, q2, is_duplicate, q1_pred, q2_pred = separate_data(train_data, pred_data)

        # Preprocess Data (Tokenize, Create embeddings (with GloVe), Add Padding)
        q1_data, q2_data, labels, q1_pred_data, q2_pred_data, word_embedding_matrix, nb_words = preprocess_data(q1, q2, is_duplicate, q1_pred, q2_pred, max_tokenized_words, embedding_dim, max_sentence_words)

        # Save processed data to files
        save_data(q1_data, q2_data, labels, q1_pred_data, q2_pred_data, word_embedding_matrix, nb_words)

    # Make Model
    model = make_model(word_embedding_matrix, nb_words, activation, loss_func, embedding_dim, dropout, optimizer, layers, max_sentence_words)

    if not model_trained:
        # Split train set
        Q1_train, Q2_train, Q1_test, Q2_test, y_train, y_test = split_train_set(q1_data, q2_data, labels)

        # Train Model
        callbacks = [ModelCheckpoint('question_pairs_weights.h5', monitor='val_accuracy', save_best_only=True)]
        history = model.fit([Q1_train, Q2_train], y_train, epochs=epochs, validation_split=0.1, verbose=2, batch_size=batch_size, callbacks=callbacks)

        # Evaluate Model
        evaluate(model, Q1_test, Q2_test, y_test)

    else:
        # Make predictions
        predictions = predict(model, q1_pred_data, q2_pred_data)

        # Save them in a file
        save_predictions(predictions)

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
from tensorflow.python.keras.layers import Embedding, TimeDistributed, Lambda, Dense, Dropout, BatchNormalization
from tensorflow.python.keras.utils.data_utils import get_file
import numpy as np


def separate_data(dataset):
    print('Separating Data')
    q1 = dataset.Question1.tolist()
    q2 = dataset.Question2.tolist()
    is_dup = dataset.IsDuplicate.tolist()

    return q1, q2, is_dup


def read_data(rows):
    print('Reading Data')
    train_data = pandas.read_csv("datasets2020/datasets/q2b/train.csv", delimiter=',', nrows=rows)
    test_data = pandas.read_csv("datasets2020/datasets/q2b/test_without_labels.csv", delimiter=',', nrows=rows)
    return train_data, test_data


def tokenize(q1, q2, max_tokenized_words):
    print('Tokenizing Data')
    questions = q1 + q2
    tokenizer = Tokenizer(num_words=max_tokenized_words)
    tokenizer.fit_on_texts(questions)
    question1_word_sequences = tokenizer.texts_to_sequences(q1)
    question2_word_sequences = tokenizer.texts_to_sequences(q2)
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
    labels = np.array(is_dup, dtype=int)

    return q1_data, q2_data, labels


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


def preprocess_data(q1, q2, is_duplicate, max_tokenized_words, embedding_dim, max_sentence_words):
    # Tokenize sentences into words
    question1_word_sequences, question2_word_sequences, word_index = tokenize(q1, q2, max_tokenized_words)

    # Create an embeddings dictionary from Glove embeddings
    embeddings_index = process_glove()

    # Create word embedding matrix according to Glove embeddings and our dataset
    word_embedding_matrix, nb_words = embedding_matrix(word_index, embeddings_index, embedding_dim, max_tokenized_words)

    # Insert zero paddings
    q1_data, q2_data, labels = zero_pad(question1_word_sequences, question2_word_sequences, is_duplicate, max_sentence_words)

    return q1_data, q2_data, labels, word_embedding_matrix, nb_words


def split_train_set(q1_data, q2_data, labels):
    q_data = np.stack((q1_data, q2_data), axis=1)
    x_train, x_test, y_train, y_test = train_test_split(q_data, labels, test_size=0.3, random_state=55964338)
    Q1_train = x_train[:, 0]
    Q2_train = x_train[:, 1]
    Q1_test = x_test[:, 0]
    Q2_test = x_test[:, 1]

    return Q1_train, Q2_train, Q1_test, Q2_test, y_train, y_test


def make_embeddings(word_embedding_matrix, nb_words, embedding_dim, max_sentence_words):
    question = Input(shape=(max_sentence_words,))

    q = Embedding(nb_words + 1, embedding_dim, weights=[word_embedding_matrix], input_length=max_sentence_words, trainable=False)(question)
    q = TimeDistributed(Dense(embedding_dim, activation='relu'))(q)
    q = Lambda(lambda x: backend.max(x, axis=1), output_shape=(embedding_dim,))(q)

    return question, q


def make_model(word_embedding_matrix, nb_words, activation, loss_func, embedding_dim, dropout, optimizer, layers, max_sentence_words):
    print('Creating the model')
    #
    question1, q1 = make_embeddings(word_embedding_matrix, nb_words, embedding_dim, max_sentence_words)
    question2, q2 = make_embeddings(word_embedding_matrix, nb_words, embedding_dim, max_sentence_words)

    #
    merged = concatenate([q1, q2])

    #
    for _ in range(layers):
        merged = Dense(200, activation=activation)(merged)
        merged = Dropout(dropout)(merged)
        merged = BatchNormalization()(merged)

    #
    is_duplicate = Dense(1, activation='sigmoid')(merged)

    model = Model(inputs=[question1, question2], outputs=is_duplicate)
    model.compile(loss=loss_func, optimizer=optimizer, metrics=['accuracy'])

    return model


def evaluate(model, Q1_test, Q2_test, y_test):
    model.load_weights('question_pairs_weights.h5')
    loss, accuracy = model.evaluate([Q1_test, Q2_test], y_test, verbose=0)
    print('Test loss = {0:.4f}, test accuracy = {1:.4f}'.format(loss, accuracy))


def save(q1_data, q2_data, labels, word_embedding_matrix, nb_words):
    np.save(open('q1_train.npy', 'wb'), q1_data)
    np.save(open('q2_train.npy', 'wb'), q2_data)
    np.save(open('label_train.npy', 'wb'), labels)
    np.save(open('word_embedding_matrix.npy', 'wb'), word_embedding_matrix)
    with open('nb_words.json', 'w') as f:
        json.dump({'nb_words': nb_words}, f)


if __name__ == '__main__':
    # Parameters
    rows = 10000
    embedding_dim = 300
    max_tokenized_words = 200000
    dropout = 0.1
    optimizer = 'adam'
    layers = 4
    activation = 'relu'
    loss_func = 'binary_crossentropy'
    epochs = 25
    batch_size = 32
    max_sentence_words = 40
    data_processed = 1

    if data_processed:
        # Load Data
        q1_data = np.load(open('q1_train.npy', 'rb'))
        q2_data = np.load(open('q2_train.npy', 'rb'))
        labels = np.load(open('label_train.npy', 'rb'))
        word_embedding_matrix = np.load(open('word_embedding_matrix.npy', 'rb'))
        with open('nb_words.json', 'r') as f:
            nb_words = json.load(f)['nb_words']
    else:
        # Read Dataset
        train_data, test_data = read_data(rows)

        # Separate Train set into Question1 column, Question2 column, Labels column
        q1, q2, is_duplicate = separate_data(train_data)

        # Preprocess Data (Tokenize, Create embeddings (GloVe), Padding)
        q1_data, q2_data, labels, word_embedding_matrix, nb_words = preprocess_data(q1, q2, is_duplicate, max_tokenized_words, embedding_dim, max_sentence_words)

        # Save processed data to files
        save(q1_data, q2_data, labels, word_embedding_matrix, nb_words)

    # Split train set
    Q1_train, Q2_train, Q1_test, Q2_test, y_train, y_test = split_train_set(q1_data, q2_data, labels)

    # Make Model
    model = make_model(word_embedding_matrix, nb_words, activation, loss_func, embedding_dim, dropout, optimizer, layers, max_sentence_words)

    # Train Model
    callbacks = [ModelCheckpoint('question_pairs_weights.h5', monitor='val_accuracy', save_best_only=True)]
    history = model.fit([Q1_train, Q2_train], y_train, epochs=epochs, validation_split=0.3, verbose=2, batch_size=batch_size, callbacks=callbacks)

    # Evaluate Model
    evaluate(model, Q1_test, Q2_test, y_test)

    print('The End')

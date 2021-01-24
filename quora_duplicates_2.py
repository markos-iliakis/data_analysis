import csv

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import Input
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.layers import Lambda, concatenate, BatchNormalization, Dense, Dropout
from tensorflow.python.keras.models import Model
from sklearn.metrics import classification_report

tf.compat.v1.disable_eager_execution()
module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
embed = hub.Module(module_url)


def separate_data(dataset, pred_data):
    print('Separating Data')

    dataset.dropna(inplace=True)
    q1 = dataset.Question1.tolist()
    q2 = dataset.Question2.tolist()
    is_dup = dataset.IsDuplicate.tolist()

    q1_pred = pred_data.Question1.tolist()
    q2_pred = pred_data.Question2.tolist()

    return q1, q2, is_dup, q1_pred, q2_pred


def read_data(rows):
    print('Reading Data')
    train_data = pd.read_csv("datasets2020/datasets/q2b/train.csv", delimiter=',', nrows=rows)
    test_data = pd.read_csv("datasets2020/datasets/q2b/test_without_labels.csv", delimiter=',', nrows=rows)
    return train_data, test_data


def universal_embedding(x):
    return embed(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["default"]


def apply_embeddings_2():
    # Taking the question as input and creating an embedding for each question before feed it to neural network
    q = Input(shape=(1,), dtype=tf.string)
    embedding_q = Lambda(universal_embedding, output_shape=(512,))(q)

    return q, embedding_q


def make_model_2(activation, loss_func, dropout, optimizer, layers):
    print('Creating the model')

    # Create Tensors and integrate the embeddings
    q1, embedding_q1 = apply_embeddings_2()
    q2, embedding_q2 = apply_embeddings_2()

    # Concatenating both input layers
    merged = concatenate([embedding_q1, embedding_q2])

    # Normalizing the input layer,applying dense and dropout  layer for fully connected model and to avoid overfitting
    for _ in range(layers):
        merged = BatchNormalization()(merged)
        merged = Dense(200, activation=activation)(merged)
        merged = Dropout(dropout)(merged)

    # Using the Sigmoid as the activation function and binary crossentropy for binary classification as 0 or 1
    merged = BatchNormalization()(merged)
    pred = Dense(2, activation='sigmoid')(merged)
    model = Model(inputs=[q1, q2], outputs=pred)
    model.compile(loss=loss_func, optimizer=optimizer, metrics=['accuracy'])

    return model


def evaluate(model, Q1_test, Q2_test, y_test):
    model.load_weights('question_pairs_weights_2.h5')
    loss, accuracy = model.evaluate([Q1_test, Q2_test], y_test, verbose=0)
    print('Test loss = {0:.4f}, test accuracy = {1:.4f}'.format(loss, accuracy))


def predict(model, q1_pred_data, q2_pred_data, y_test=None):
    # Load trained weights
    model.load_weights('question_pairs_weights_2.h5')

    # Make predictions
    predictions = model.predict([q1_pred_data, q2_pred_data])

    # Generate classes from predictions
    duplicates = list()
    for pred in predictions:
        dup = 0
        if pred[0] > 0.5:
            dup = 1
        duplicates.append(dup)

    if y_test:
        # Print Metrics
        print(classification_report(y_test, duplicates))

    return duplicates


def save_predictions(predictions):
    ids = [i for i in range(283003, 283003 + len(predictions))]

    with open('results_duplicates_2.csv', 'w') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(("Id", "Predicted"))
        wr.writerows(zip(ids, predictions))


if __name__ == '__main__':
    # Parameters
    rows = None
    dropout = 0.1
    optimizer = 'adam'
    layers = 4
    activation = 'relu'
    loss_func = 'binary_crossentropy'
    epochs = 10
    max_sentence_words = 40
    batch_size = 512
    model_trained = 1

    # Read Dataset
    train_data, pred_data = read_data(rows)

    # Separate Train set into Question1 column, Question2 column, Labels column
    q1, q2, is_duplicate, q1_pred, q2_pred = separate_data(train_data, pred_data)

    # Make the model
    model = make_model_2(activation, loss_func, dropout, optimizer, layers)

    # Split Train set
    Q1_train, Q1_test, Q2_train, Q2_test, y_train, y_test = train_test_split(q1, q2, is_duplicate, test_size=0.2, random_state=42)

    #
    train_q1 = np.array(Q1_train, dtype=object)[:, np.newaxis]
    train_q2 = np.array(Q2_train, dtype=object)[:, np.newaxis]
    test_q1 = np.array(Q1_test, dtype=object)[:, np.newaxis]
    test_q2 = np.array(Q2_test, dtype=object)[:, np.newaxis]
    train_labels = np.asarray(pd.get_dummies(y_train), dtype=np.int8)
    test_labels = np.asarray(pd.get_dummies(y_test), dtype=np.int8)

    if not model_trained:
        with tf.compat.v1.Session() as session:
            set_session(session)
            session.run(tf.compat.v1.global_variables_initializer())
            session.run(tf.compat.v1.tables_initializer())
            # Train Model
            callbacks = [
                ModelCheckpoint('question_pairs_weights_2.h5', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=True, mode='auto', save_freq=1)]
            history = model.fit([train_q1, train_q2], train_labels, validation_data=([test_q1, test_q2], test_labels), epochs=epochs, batch_size=batch_size, callbacks=callbacks)

    # Evaluate Model
    # evaluate(model, test_q1, test_q2, test_labels)
    predict(model, test_q1, test_q2, test_labels)

    # Make predictions
    q1_pred_data = np.array(q1_pred, dtype=object)[:, np.newaxis]
    q2_pred_data = np.array(q2_pred, dtype=object)[:, np.newaxis]
    predictions = predict(model, q1_pred_data, q2_pred_data)

    # Save them in a file
    save_predictions(predictions)

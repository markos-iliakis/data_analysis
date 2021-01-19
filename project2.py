import pandas
from sklearn.metrics.pairwise import cosine_similarity
from datasketch import MinHash, MinHashLSH
from lshashpy3 import LSHash
import time
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import nltk


def preprocess_data(train_data, test_data, perms):
    # Set LSH preprocessing timer
    start_build_time = time.time()

    # Create MniHashes
    train_sets = dict()
    for index, row in train_data.iterrows():
        train_sets[row.Id] = MinHash(num_perm=perms)
        for word in row.Content:
            train_sets[row.Id].update(word.encode('utf8'))

    test_sets = dict()
    for index, row in test_data.iterrows():
        test_sets[row.Id] = MinHash(num_perm=perms)
        for word in row.Content:
            test_sets[row.Id].update(word.encode('utf8'))

    # Create LSH index
    lsh = MinHashLSH(threshold=0.8, num_perm=perms)
    for key in train_sets.keys():
        lsh.insert(key, train_sets[key])

    # Stop LSH preprocessing timer
    stop_build_time = time.time()

    return stop_build_time - start_build_time, lsh, test_sets


def query_lsh(lsh, test_sets):
    # Set LSH query timer
    start_query_time = time.time()

    # Query test set
    duplicates = 0
    for key in test_sets.keys():
        if len(lsh.query(test_sets[key])) > 0:
            duplicates += 1

    # Stop LSH query timer
    stop_query_time = time.time()

    return stop_query_time - start_query_time, duplicates


def get_jaccard_sim(str1, str2):
    a = set(str1.split())
    b = set(str2.split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def exact_duplicates_jaccard(train_data, test_data):
    # Set Jaccard query timer
    start_jaccard_time = time.time()

    # Exact Jaccard Similarity
    duplicates = 0
    for _, test_row in test_data.iterrows():
        for _, train_row in train_data.iterrows():
            distance = get_jaccard_sim(test_row.Content, train_row.Content)
            if distance > 0.8:
                duplicates += 1
                break

    # Stop Jaccard query timer
    stop_jaccard_time = time.time()

    return duplicates, stop_jaccard_time - start_jaccard_time


def vectorize_data(train_data, test_data):

    count_vectorizer = CountVectorizer(analyzer="word")

    train_texts = []
    for _, train_row in train_data.iterrows():
        train_texts.append(train_row.Content)

    test_texts = []
    for _, test_row in test_data.iterrows():
        test_texts.append(test_row.Content)

    vec_data = count_vectorizer.fit_transform(train_texts + test_texts)

    return vec_data, len(train_texts)


def exact_duplicates_cosine(train_data, test_data):
    # Set cosine query time
    start_cosine_time = time.time()

    # Vectorize train and text texts
    vec_data, train_size = vectorize_data(train_data, test_data)

    # Exact cosine similarity
    duplicates = 0
    for test_row in range(train_size, vec_data.shape[0]):
        for train_row in range(0, train_size):
            distance = cosine_similarity(vec_data[test_row, :], vec_data[train_row, :])
            if distance > 0.8:
                duplicates += 1
                break

    # Stop cosine query time
    stop_cosine_time = time.time()

    return stop_cosine_time - start_cosine_time, duplicates


def lsh_cosine(train_data, test_data, k, l=1):

    vec_data, train_size = vectorize_data(train_data, test_data)

    d = vec_data.shape[1]

    # Create lsh
    start_build_time = time.time()
    lsh = LSHash(hash_size=k, input_dim=d, num_hashtables=l)

    # Insert data
    for row in range(0, train_size):
        lsh.index(vec_data[row, :].toarray()[0])

    stop_build_time = time.time()

    # TODO return only test vectorizes data instead of vec_data, train_size
    return lsh, stop_build_time - start_build_time, vec_data, train_size


# TODO arguments: lsh, test_data
def query_lsh_cosine(lsh, vec_data, train_size):
    start_query_time = time.time()
    duplicates = 0
    for row in range(train_size, vec_data.shape[0]):
        nn = lsh.query(vec_data[row, :].toarray()[0], distance_func="cosine")
        for _, distance in nn:
            if distance > 0.8:
                duplicates += 1
                break
    stop_query_time = time.time()

    return stop_query_time-start_query_time, duplicates


def read_data(rows):
    train_data = pandas.read_csv("datasets2020/datasets/q2a/corpusTrain.csv", delimiter=',', nrows=rows)
    test_data = pandas.read_csv("datasets2020/datasets/q2a/corpusTest.csv", delimiter=',', nrows=rows)
    return train_data, test_data


def display_results(method, duplicates, query_time, build_time=0.0):
    print(method, 'Duplicate sentences : ', duplicates, ' Build time : ', build_time, 's Query Time : ', query_time, 's')


if __name__ == '__main__':
    # Number of permutations
    perms = 32  # 16 | 32 | 64
    rows = 200  # Number of rows to read

    # Read data
    train_data, test_data = read_data(rows)

    # Preprocess Data for Minhash lsh
    build_time, lsh, test_sets = preprocess_data(train_data, test_data, perms)

    # Query Data in Minhash lsh
    query_time, duplicates = query_lsh(lsh, test_sets)

    # Display results
    display_results("Minhash lsh", duplicates, query_time, build_time)

    # Find exact duplicates with Jaccard
    duplicates, query_time = exact_duplicates_jaccard(train_data, test_data)

    # Display results
    display_results("Exact jaccard", duplicates, query_time)

    # Find exact duplicates with cosine similarity
    query_time, duplicates = exact_duplicates_cosine(train_data, test_data)

    # Display results
    display_results("Exact cosine", duplicates, query_time)

    # Build lsh
    lsh, build_time, vec_data, train_size = lsh_cosine(train_data, test_data, k=8)

    #  Find duplicates with lsh-cosine
    query_time, duplicates = query_lsh_cosine(lsh, vec_data, train_size)

    display_results("lsh-cosine", duplicates, query_time, build_time)


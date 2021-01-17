import pandas
import sklearn.metrics.pairwise
from datasketch import MinHash, MinHashLSH
from lshashpy3 import LSHash
import time
import numpy as np


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
    for index, test_row in test_data.iterrows():
        for index2, train_row in train_data.iterrows():
            distance = get_jaccard_sim(test_row.Content, train_row.Content)
            if distance > 0.8:
                duplicates += 1
                break

    # Stop Jaccard query timer
    stop_jaccard_time = time.time()

    return duplicates, stop_jaccard_time - start_jaccard_time


def read_data(rows):
    train_data = pandas.read_csv("datasets2020/datasets/q2a/corpusTrain.csv", delimiter=',', nrows=rows)
    test_data = pandas.read_csv("datasets2020/datasets/q2a/corpusTest.csv", delimiter=',', nrows=rows)
    return train_data, test_data


def display_results(duplicates, build_time, query_time, exact_duplicates, jaccard_query_time):
    print('MinHashLSH duplicate sentences : ', duplicates, ' Build time : ', build_time, 's Query Time : ', query_time, 's')
    print('Exact duplicate sentences : ', exact_duplicates, 'Query Time : ', jaccard_query_time, 's')


if __name__ == '__main__':
    # Number of permutations
    perms = 32  # 16 | 32 | 64
    rows = 500  # Number of rows to read

    # Read data
    train_data, test_data = read_data(rows)

    # Preprocess Data
    # build_time, lsh, test_sets = preprocess_data(train_data, test_data, perms)

    # Query Data
    # query_time, duplicates = query_lsh(lsh, test_sets)

    # Find exact duplicates with Jaccard
    # exact_duplicates, jaccard_query_time = exact_duplicates_jaccard(train_data, test_data)

    # Display results
    # display_results(duplicates, build_time, query_time, exact_duplicates, jaccard_query_time)

    # k = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    k = 5
    l = 1

    d = max([len(row.Content) for _, row in train_data.iterrows()])

    # Create lsh
    lsh = LSHash(hash_size=k, input_dim=d, num_hashtables=l)

    # Insert data
    for _, row in train_data.iterrows():
        lsh.index(row.Content, extra_data=row.Id)

    results = dict()
    for _, row in test_data.iterrows():
        nn = lsh.query(row.Content, distance_func="cosine")
        results[row.Id] = nn

    print(results)

# sklearn.metrics.pairwise.cosine_similarity(X, Y=None, dense_output=True)

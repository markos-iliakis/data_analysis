import pandas
from sklearn.metrics.pairwise import cosine_similarity
from datasketch import MinHash, MinHashLSH
from lshashpy3 import LSHash
import time
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from nltk import word_tokenize
from gensim.models.word2vec import Word2Vec
import nltk


def train_min_hash_lsh(train_data, test_data, perms):
    print('Train minHashLSH')

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
    print('Query minHashLSH')

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


def vectorize_data_with_bow(train_data, test_data):

    count_vectorizer = CountVectorizer(analyzer="word")

    train_texts = []
    for _, train_row in train_data.iterrows():
        train_texts.append(train_row.Content)

    test_texts = []
    for _, test_row in test_data.iterrows():
        test_texts.append(test_row.Content)

    train_data = count_vectorizer.fit_transform(train_texts)

    test_data = count_vectorizer.transform(test_texts)

    return train_data.toarray(), test_data.toarray()


def cosine_dist(x, y):
    return 1 - np.dot(x, y) / ((np.dot(x, x) * np.dot(y, y)) ** 0.5)


def exact_duplicates_cosine(train_set, test_set):
    # Set cosine query time
    start_cosine_time = time.time()

    # Vectorize train and text texts
    train_data, test_data = vectorize_data_with_bow(train_set, test_set)

    # Exact cosine similarity
    duplicates = 0
    for test_row in test_data:
        for train_row in train_data:
            distance = cosine_dist(test_row, train_row)
            if distance < 0.2:
                duplicates += 1
                break

    # Stop cosine query time
    stop_cosine_time = time.time()

    return stop_cosine_time - start_cosine_time, duplicates


def lsh_cosine(train_data, test_data, k, l=1):

    train_data, test_data = vectorize_data_with_bow(train_data, test_data)

    d = train_data.shape[1]

    # Create lsh
    start_build_time = time.time()
    lsh = LSHash(hash_size=k, input_dim=d, num_hashtables=l)

    # Insert data
    for train_row in train_data:
        lsh.index(train_row)

    stop_build_time = time.time()

    return lsh, stop_build_time - start_build_time, test_data


# TODO arguments: lsh, test_data
def query_lsh_cosine(lsh, test_data):
    start_query_time = time.time()
    duplicates = 0
    for test_row in test_data:
        nn = lsh.query(test_row, distance_func="cosine")
        for _, distance in nn:
            if distance < 0.2:
                duplicates += 1
                break
    stop_query_time = time.time()

    return stop_query_time-start_query_time, duplicates


def read_data(rows):
    print('Reading Data')
    path = './datasets2020/datasets/q2a/'
    train_data = pandas.read_csv(path + "corpusTrain.csv", delimiter=',', nrows=5000)
    test_data = pandas.read_csv(path + "corpusTest.csv", delimiter=',', nrows=None)
    return train_data, test_data


def display_results(method, duplicates, query_time, build_time=0.0):
    print(method, 'Duplicate sentences : ', duplicates, ' Build time : ', build_time, 's Query Time : ', query_time, 's')


def average_words_embeddings(words, word2vec):
    word_vec = []
    for word in words:
        try:
            word_vec.append(word2vec.wv[word])
        except KeyError:
            print(word)

    return np.mean(np.array(word_vec), axis=0)


def vec_with_Word2Vec(train_set, test_set, sg=0):
    # Vectorize train_set...............................................................................................
    sentences = []
    for _, row in train_set.iterrows():
        sentences.append(word_tokenize(row.Content))
    if test_set is not None:
        for _, row in test_set.iterrows():
            sentences.append(word_tokenize(row.Content))

    model = Word2Vec(sentences=sentences, min_count=0, size=100, window=5, sg=sg)

    x_train_content = []
    for _, row in train_set.iterrows():
        x_train_content.append(average_words_embeddings(word_tokenize(row.Content), model))
    x_train_content = np.array(x_train_content)

    # Vectorize test_set................................................................................................
    if test_set is not None:
        x_test_content = []
        for _, row in test_set.iterrows():
            x_test_content.append(average_words_embeddings(word_tokenize(row.Content), model))
        x_test_content = np.array(x_test_content)
    else:
        x_test_content = None

    return x_train_content, x_test_content


if __name__ == '__main__':
    # Number of permutations
    perms = 64  # 16 | 32 | 64
    rows = 3  # Number of rows to read

    # Read data
    train_data, test_data = read_data(rows)

    # # Preprocess Data for Minhash lsh
    build_time, lsh, test_sets = train_min_hash_lsh(train_data, test_data, perms)

    # # Query Data in Minhash lsh
    query_time, duplicates = query_lsh(lsh, test_sets)

    # # Display results
    display_results("Minhash lsh", duplicates, query_time, build_time)

    # Find exact duplicates with Jaccard
    duplicates, query_time = exact_duplicates_jaccard(train_data, test_data)

    # Display results
    display_results("Exact jaccard", duplicates, query_time)

    # Find exact duplicates with cosine similarity
    query_time, duplicates = exact_duplicates_cosine(train_data, test_data)

    # Display results
    display_results("Exact cosine", duplicates, query_time)

    # # Build lsh
    lsh, build_time, vec_test_data = lsh_cosine(train_data, test_data, k=2)

    #  Find duplicates with lsh-cosine
    query_time, duplicates = query_lsh_cosine(lsh, vec_test_data)

    display_results("lsh-cosine", duplicates, query_time, build_time)


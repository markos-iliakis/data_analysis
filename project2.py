import pandas
from datasketch import MinHash, MinHashLSH
from lshashpy3 import LSHash

if __name__ == '__main__':
    train_data = pandas.read_csv("datasets2020/datasets/q2a/corpusTrain.csv", delimiter=',')
    test_data = pandas.read_csv("datasets2020/datasets/q2a/corpusTest.csv", delimiter=',')

    # Create LSH index
    # lsh = MinHashLSH(threshold=0.8, num_perm=128)
    # lsh.insert("m2", m2)
    # lsh.insert("m3", m3)
    # result = lsh.query(m1)
    # print("Approximate neighbours with Jaccard similarity > 0.5", result)

    # create 6-bit hashes for input data of 8 dimensions:
    lsh = LSHash(6, 8)

    # index vector
    lsh.index([2, 3, 4, 5, 6, 7, 8, 9])

    # index vector and extra data
    lsh.index([10, 12, 99, 1, 5, 31, 2, 3], extra_data="vec1")
    lsh.index([10, 11, 94, 1, 4, 31, 2, 3], extra_data="vec2")

    # query a data point
    top_n = 1
    nn = lsh.query([1, 2, 3, 4, 5, 6, 7, 7], num_results=top_n, distance_func="euclidean")
    print(nn)

    # unpack vector, extra data and vectorial distance
    top_n = 3
    nn = lsh.query([10, 12, 99, 1, 5, 30, 1, 1], num_results=top_n, distance_func="euclidean")
    for ((vec, extra_data), distance) in nn:
        print(vec, extra_data, distance)
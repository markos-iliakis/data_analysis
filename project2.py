import pandas
from datasketch import MinHash, MinHashLSH
from lshashpy3 import LSHash

if __name__ == '__main__':

    # Number of permutations
    perms = 16  # 16 | 32 | 64

    train_data = pandas.read_csv("datasets2020/datasets/q2a/corpusTrain.csv", delimiter=',')
    test_data = pandas.read_csv("datasets2020/datasets/q2a/corpusTest.csv", delimiter=',')

    # Create MniHashes
    train_sets = dict()
    for row in train_data.iteritems():
        train_sets[row.Id] = MinHash(num_perm=perms)
        for word in row.Content:
            train_sets[row.Id].update(word.encode('utf8'))

    test_sets = dict()
    for row in test_data.iteritems():
        test_sets[row.Id] = MinHash(num_perm=perms)
        for word in row.Content:
            test_sets[row.Id].update(word.encode('utf8'))

    # Create LSH index
    lsh = MinHashLSH(threshold=0.8, num_perm=perms)
    for key in train_sets.keys():
        lsh.insert(key, train_sets[key])

    # Query test set
    for key in test_sets.keys():
        result = lsh.query(test_sets[key])
        print("Approximate neighbours with Jaccard similarity > 0.5", result)

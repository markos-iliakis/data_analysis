import pandas
import numpy as np
import wordcloud as wc
import nltk
from nltk import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords as e_stopwords
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, articles):
        return [self.wnl.lemmatize(t) for t in word_tokenize(articles)]


def stop_words_finder(data, stop_words_num):
    count_vectorizer = []
    categ = set(data['Category'])

    ## Find the most dominant words in each category ##
    i = 0
    for x in categ:
        count_vectorizer.append(CountVectorizer(stop_words=ENGLISH_STOP_WORDS, max_features=stop_words_num))
        res = count_vectorizer[i].fit_transform(data.loc[data['Category'] == x, 'Content'])
        i = i + 1

    ## All the sets intersected with the first one ##
    result = set(count_vectorizer[0].vocabulary_.keys())
    for set_num in range(1, i):
        result = set(result).intersection(set(count_vectorizer[set_num].vocabulary_.keys()))

    return result


def make_wordclouds(stopwords, labels, texts):
    # Make the wordclouds
    for label in labels:
        wc.WordCloud(max_words=1000, mask=None, stopwords=stopwords, margin=10, random_state=1).generate(
            texts[label]).to_file("./wordclouds/" + label + ".png")


def preprocess_data(train_data):
    # Remove stop words
    stopwords = set(e_stopwords.words('english'))

    # Find all different categories
    labels = set(train_data.Label)

    # Separate categories
    texts = {key: train_data[train_data.Label == key].Content.tolist() for key in labels}

    # Flatten the lists
    for key in texts.keys():
        text = ""
        l = [text for sublist in texts[key] for text in sublist]
        for t in l:
            text += t
        texts[key] = text

    return stopwords, labels, texts


if __name__ == '__main__':
    # Read data set
    train_data = pandas.read_csv("datasets2020/datasets/q1/train.csv", delimiter=',')

    # Preprocess Data
    stopwords, labels, texts = preprocess_data(train_data)

    # make_wordclouds(stopwords, labels, texts)

    # Make the vectorization
    vectorizer = TfidfVectorizer(stop_words=stopwords)
    X = vectorizer.fit_transform(train_data.Content)

    # Make the SVD representation
    svd_obj = TruncatedSVD(n_components=250)
    svd = svd_obj.fit_transform(X)

    # Make Bag of Words representation
    count_vectorizer = CountVectorizer(analyzer="word", tokenizer=nltk.word_tokenize, preprocessor=None, stop_words=stopwords , max_features=None)
    bag_of_words = count_vectorizer.fit_transform(train_data.Content)

    # normalizer = Normalizer(copy=False)
    # lsa = make_pipeline(count_vectorizer, svd_obj, normalizer)
    # X_train_content = lsa.fit_transform(texts['Content'] + texts['Title'])
    # tests = lsa.fit_transform(test['Content'] + test['Title'])

    print("h")
import pandas
import wordcloud as wc
import nltk
from nltk import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords as e_stopwords
from sklearn import svm
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import make_scorer, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer


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


def make_wordclouds(stopwords, labels, data_set):
    # Separate categories
    texts = {key: data_set[data_set.Label == key].Content.tolist() for key in labels}

    # Flatten the lists
    for key in texts.keys():
        text = ""
        l = [text for sublist in texts[key] for text in sublist]
        for t in l:
            text += t
        texts[key] = text

    # Make the wordclouds
    for label in labels:
        wc.WordCloud(max_words=1000, mask=None, stopwords=stopwords, margin=10, random_state=1).generate(
            texts[label]).to_file("./wordclouds/" + label + ".png")


def preprocess_data(data_set, vectorizer=None, representation=None, options=None):
    # Define stop words
    stopwords = set(e_stopwords.words('english'))

    # Find all different categories
    labels = set(data_set.Label)

    # Get all categories
    y = data_set.Label

    x_train_content=None

    if representation == 'svd':
        # Make the vectorization
        vec = TfidfVectorizer(stop_words=stopwords)
        x_train_content = vec.fit_transform(data_set.Content)

        # Make the SVD representation
        svd_obj = TruncatedSVD(n_components=250)
        x_train_content = svd_obj.fit_transform(x_train_content)

        # normalizer = Normalizer(copy=False)
        # count_vectorizer = CountVectorizer(stop_words=stopwords, analyzer="word", tokenizer=nltk.word_tokenize)
        # lsa = make_pipeline(count_vectorizer, repr_obj, normalizer)
        # x_train_content = lsa.fit_transform(data_set.Content + data_set.Title)

    else:
        # Make the vectorization
        vec = CountVectorizer(stop_words=stopwords, analyzer="word", tokenizer=nltk.word_tokenize)
        x_train_content = vec.fit_transform(data_set.Content)

    return stopwords, labels, y, x_train_content


def create_classifier(classifier):
    if classifier == 'svc':
        cl = svm.SVC()
        parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10, 100], 'gamma': [0.001, 0.01, 0.1]}
    else:
        cl = RandomForestClassifier()
        parameters = {'n_estimators': [10, 30, 50], 'criterion': ('gini', 'entropy'), 'min_samples_split': [2, 10]}

    return cl, parameters


def display_results(results):
    print("Precision = ", results['test_Accuracy'].mean())
    print("Recall = ", results['test_Recall'].mean())
    print("F-Measure = ", results['test_F-Measure'].mean())
    print("Accuracy = ", results['test_Accuracy'].mean())


if __name__ == '__main__':

    classifier = 'svc'  # svc (support vector machines)| rf (random forest)
    representation = 'svd'  # svd (singular value decomposition) | bow (bag of words)
    rows = 500  # Number of rows to read

    # Read data set
    data_set = pandas.read_csv("datasets2020/datasets/q1/train.csv", delimiter=',', nrows=rows)

    # Preprocess Data
    stopwords, labels, y, x_train_content = preprocess_data(data_set, representation)

    # Make Word Cloud images
    # make_wordclouds(stopwords, labels, texts)

    # Create Classifier
    cl, parameters = create_classifier(classifier)

    # Grid Search
    model = GridSearchCV(cl, parameters, refit="True")

    # Cross Validate
    results = cross_validate(model, x_train_content, y, scoring={'Precision': make_scorer(precision_score, average='micro'), 'Recall': make_scorer(recall_score, average='micro'), 'F-Measure': 'f1_micro', 'Accuracy': 'accuracy'})

    # Print Results
    display_results(results)

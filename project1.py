import pandas as pd
import wordcloud as wc
import numpy as np
import string
import spacy
from nltk import word_tokenize
from sklearn import svm
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import make_scorer, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, cross_validate, StratifiedKFold
from nltk.stem import WordNetLemmatizer
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS
from nltk.corpus import stopwords as e_stopwords

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer


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


def preprocess_train_data(train_set, test_set, algorithm , *args):

    # Preprocess train_set..............................................................................................
    # Remove punctuation
    train_set.Content = remove_punctuations(train_set.Content)

    # Lower case
    train_set.Content = lowercase(train_set.Content)

    # Lemmatization
    # spacy lemmatization
    # train_set.Content = spacy_lemmatization(train_set.Content)
    # nltk lemmatization
    train_set.Content = nltk_lemmatization(train_set.Content)

    # Remove stopwords
    stopwords = ENGLISH_STOP_WORDS.union(stop_words_finder(train_set, 500))
    train_set.Content = remove_stopwords(train_set.Content, stopwords)

    # Preprocess test_set...............................................................................................
    if test_set is not None:
        # Remove punctuation
        test_set.Content = remove_punctuations(test_set.Content)

        # Lower case
        test_set.Content = lowercase(test_set.Content)

        # Lemmatization
        # spacy lemmatization
        # train_set.Content = spacy_lemmatization(train_set.Content)
        # nltk lemmatization
        test_set.Content = nltk_lemmatization(test_set.Content)

        # Remove stopwords
        test_set.Content = remove_stopwords(test_set.Content, stopwords)

    # Vectorize dataset
    x_train_content, x_test_content = algorithm(train_set, test_set, *args)


    return stopwords, set(train_set.Label), train_set.Label, x_train_content, x_test_content


def create_classifier(classifier):
    if classifier == 'svc':
        cl = svm.SVC()
        parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10, 100], 'gamma': [0.001, 0.01, 0.1]}
    else:
        cl = RandomForestClassifier()
        parameters = {'n_estimators': [10, 30, 50], 'criterion': ('gini', 'entropy'), 'min_samples_split': [2, 10]}

    return cl, parameters


def remove_punctuations(data):
    punct = []
    punct += list(string.punctuation)
    punct += '’'
    punct.remove("'")

    punct_free_docs = []
    for _, text in data.items():
        for punctuation in punct:
            text = text.replace(punctuation, ' ')
        punct_free_docs.append(text)

    return pd.Series(punct_free_docs)


def lowercase(data):
    lowercase_doc = []
    for _, document in data.items():
        lowercase_doc.append(' '.join([token.lower() for token in word_tokenize(document)]))

    return pd.Series(lowercase_doc)


def remove_stopwords(data, stopwords):
    stopwords_free_docs = []
    for _, document in data.items():
        stopwords_free_docs.append(' '.join([token for token in word_tokenize(document) if not token in stopwords]))

    return pd.Series(stopwords_free_docs)


def nltk_lemmatization(documents):

    lemmatizer = WordNetLemmatizer()
    lemmatized_docs = []

    for _, document in documents.items():
        lemmatized_docs.append(lemmatizer.lemmatize(document))

    return pd.Series(lemmatized_docs)


def spacy_lemmatization(documents):

    nlp = spacy.load('en_core_web_lg')
    lemmatized_docs = []
    for _, document in documents.items():
        doc = nlp(document)
        lemmatized_docs.append(' '.join([token.lemma_ for token in doc]))

    return pd.Series(lemmatized_docs)


def stop_words_finder(data, stop_words_num):
    count_vectorizer = []
    categ = set(data.Label)

    # Find the most dominant words in each category
    i = 0
    for x in categ:
        count_vectorizer.append(CountVectorizer(stop_words=ENGLISH_STOP_WORDS, max_features=stop_words_num))
        res = count_vectorizer[i].fit_transform(data.loc[data.Label == x, 'Content'])
        i = i + 1

    # All the sets intersected with the first one
    result = set(count_vectorizer[0].vocabulary_.keys())
    for set_num in range(1, i):
        result = set(result).intersection(set(count_vectorizer[set_num].vocabulary_.keys()))

    return result


def vec_with_Bow(train_set, test_set):
    vec = CountVectorizer(analyzer="word", max_features=300)

    # Vectorize train_set
    x_train_content = vec.fit_transform(train_set.Content)

    # Vectorize test_set
    x_test_content = None
    if test_set is not None:
        x_test_content = vec.transform(test_set.Content)

    return x_train_content, x_test_content


def vec_with_TfIdf(train_set, test_set):
    # Vectorize train_set...............................................................................................
    vec = TfidfVectorizer()
    x_train_content = vec.fit_transform(train_set.Content)

    # Make the SVD representation
    svd_obj = TruncatedSVD(n_components=250)
    x_train_content = svd_obj.fit_transform(x_train_content)

    # Vectorize test_set................................................................................................
    x_test_content = None
    if test_set is not None:
        x_test_content = vec.transform(test_set.Content)
        x_test_content = svd_obj.transform(x_test_content)

    return x_train_content, x_test_content


def to_TaggedDoc(data_set):
    train_set = []
    for _, row in data_set.iterrows():
        tokenized_text = word_tokenize(row.Content)
        train_set.append(TaggedDocument(word_tokenize(row.Content), row.Label))

    return train_set


def vec_with_Doc2Vec(train_set, test_set, dm=1):
    # Vectorize train_set...............................................................................................
    tagged_train_set = to_TaggedDoc(train_set)

    # Doc2Vec with distributed bag of words
    model = Doc2Vec(dm=dm, vector_size=500, min_count=2, epochs=60)
    model.build_vocab(tagged_train_set)
    model.train(tagged_train_set, total_examples=model.corpus_count, epochs=model.epochs)

    x_train_content = []
    for _, row in train_set.iterrows():
        x_train_content.append(model.infer_vector(word_tokenize(row.Content)))
    x_train_content = np.array(x_train_content)

    # Vectorize test_set................................................................................................
    x_test_content = None
    if test_set is not None:
        x_test_content = []
        for _, row in test_set.iterrows():
            x_test_content.append(model.infer_vector(word_tokenize(row.Content)))
        x_test_content = np.array(x_test_content)

    return x_train_content, x_test_content


def average_words_embeddings(words, word2vec):
    word_vec = []
    for word in words:
        try:
            word_vec.append(word2vec.wv[word])
        except KeyError:
            continue

    return np.mean(np.array(word_vec), axis=0)


# sg=0 -> cbow, sg=1 -> continuous skip-gram
def vec_with_Word2Vec(train_set, test_set, sg=0):
    # Train word2vec model
    sentences = []
    for _, row in train_set.iterrows():
        sentences.append(word_tokenize(row.Content))
    if test_set is not None:
        for _, row in test_set.iterrows():
            sentences.append(word_tokenize(row.Content))

    model = Word2Vec(sentences=sentences, min_count=2, size=500, window=5, sg=sg)

    # Vectorize train_set...............................................................................................
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


def display_results(results):
    print("Precision = ", results['test_Accuracy'].mean())
    print("Recall = ", results['test_Recall'].mean())
    print("F-Measure = ", results['test_F-Measure'].mean())
    print("Accuracy = ", results['test_Accuracy'].mean())


if __name__ == '__main__':

    vec_algorithm = vec_with_Bow
    classifier = 'svc'  # svc (support vector machines)| rf (random forest)
    rows = 5000  # Number of rows to read

    # Read train dataset
    train_set = pd.read_csv("train.csv", delimiter=',', nrows=rows)

    # FOR TESTING
    test_set = pd.read_csv("test_without_labels.csv", delimiter=',')

    # Preprocess train data
    stopwords, labels, y, x_train_content, x_test_content = preprocess_train_data(train_set=train_set, test_set=test_set, algorithm=vec_algorithm)
    print("end of preprocessing")

    # # Make Word Cloud images
    make_wordclouds(stopwords, labels, train_set)

    # # Create Classifier
    cl, parameters = create_classifier(classifier)
    #
    # # Grid Search
    model = GridSearchCV(cl, parameters, refit="True")
    #
    # # FOR TRAINING
    # # Cross Validate
    # kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
    # results = cross_validate(model, x_train_content, y, cv=kf, scoring={'Precision': make_scorer(precision_score, average='micro'), 'Recall': make_scorer(recall_score, average='micro'), 'F-Measure': 'f1_micro', 'Accuracy': 'accuracy'})
    # # Print Results
    # display_results(results)

    # FOR TESTING
    model.fit(x_train_content, y)
    y_pred = model.predict(x_test_content)

    d = {'Id': test_set.Id, 'Predicted': y_pred}
    df = pd.DataFrame(data=d)
    df.to_csv('testSet_categories.csv', encoding='utf-8', sep='\t', index=False)

import pickle
from tf_tfidf_vectorizer import create_vectorizer_tfidf, create_vectorizer_tf, parts_dir_path, stop_words
from tf_tfidf_vectorizer import tf_tfidf_result, top_idf_features, top_mean_feats
import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold
import numpy as np

model_dir_path = os.path.dirname(os.path.realpath(__file__))
train_books = pd.read_csv(model_dir_path + "\\..\\dataset\\train.csv")


# np.set_printoptions(threshold=np.nan)
# Parameters to change:
# name_train_books: names of files used for training
# name_test_books: names of files used for testing
# MAX_DF
# MIN_DF
# TOP_N_WORDS: in case "max_features" paramter is used in tfidf vectorizer (only if use_max_features is True)
# STOP_WORDS
# TFIDF : True: use TFIDF
#         False: use TF
# VOC_IDF : True: voc from best IDF if TFIDF = True. If TFIDF = False, VOC_IDF is ignored
#           False: if TFIDF=True, voc from best average tfidf
# use_max_features - bool, if true: use "max_features" parameter of tfidf vectorizer
# norm = norm parameter for tfidf vectorizer
def find_feature_vectors(name_train_books, name_test_books, MAX_DF = 1.0, TOP_N_WORDS=30000, MIN_DF = 0.0, STOP_WORDS=None, TFIDF=False,
                            VOC_IDF=False, use_max_features=True, norm='l1', ngram=(1,1), sublinear_tf=False):
    if TFIDF == False:
        if use_max_features == True:
            vectorizer, train_matrix = create_vectorizer_tf(name_train_books, max_features=TOP_N_WORDS, in_stop_words=STOP_WORDS,
                                                              in_max_df=MAX_DF, in_min_df=MIN_DF, norm=norm, ngram=ngram,
                                                              sublinear_tf=sublinear_tf)
        else:
            vectorizer, train_matrix = create_vectorizer_tf(name_train_books, in_stop_words=STOP_WORDS,
                                                            in_max_df=MAX_DF, in_min_df=MIN_DF, norm=norm, ngram=ngram,
                                                            sublinear_tf=sublinear_tf)
            # find the vocabulary by taking the TOP_N_WORDS words with highest tf
            vocabulary = tf_tfidf_result(vectorizer, train_matrix, in_top_n=TOP_N_WORDS, out_file=False)
            # # train a new vectorizer with the found vocabulary
            vectorizer, train_matrix = create_vectorizer_tf(name_train_books, vocab=vocabulary['feature'],
                                                            in_stop_words=STOP_WORDS, norm=norm, ngram=ngram,
                                                            sublinear_tf=sublinear_tf)
    else:
        if use_max_features == True:
            vectorizer, train_matrix = create_vectorizer_tfidf(name_train_books, max_features=TOP_N_WORDS, in_stop_words=STOP_WORDS,
                                                              in_max_df=MAX_DF, in_min_df=MIN_DF, norm=norm, ngram=ngram,
                                                              sublinear_tf=sublinear_tf)
        else:
            vectorizer, train_matrix = create_vectorizer_tfidf(name_train_books, in_stop_words=STOP_WORDS,
                                                                in_max_df=MAX_DF, in_min_df=MIN_DF, norm=norm, ngram=ngram,
                                                               sublinear_tf=sublinear_tf)

            if VOC_IDF:
                # find the vocabulary by taking the 30,000 words with highest idf
                top_words = top_idf_features(vectorizer, top_n=TOP_N_WORDS)
            else:
                # find the vocabulary by taking the 30,000 words with highest tfidf (average)
                top_words = tf_tfidf_result(vectorizer, train_matrix, in_top_n=TOP_N_WORDS, out_file=False)
                top_words = top_words['feature']
            vocabulary = {}
            i = 0
            for top_word in top_words:
                vocabulary[top_word] = i
                i += 1
            # train a new vectorizer with the found vocabulary
            vectorizer, train_matrix = create_vectorizer_tfidf(name_train_books, vocab=vocabulary,
                                                           in_stop_words=STOP_WORDS, norm=norm, ngram=ngram,
                                                           sublinear_tf=sublinear_tf)

    test_texts = []
    for file in name_test_books:
        file_opened = open(parts_dir_path + "\\" + file, "r", encoding="ISO-8859-1")
        file_opened = file_opened.read()
        test_texts.append(file_opened)

    test_matrix = vectorizer.transform(test_texts)

    return train_matrix, test_matrix


def get_tfidf_manual_dict (indices_list, in_stop_words=None, max_features=20000, in_max_df=1.0, in_min_df=0.0):
    x_train = train_books.loc[:, train_books.columns != 'Year']
    matrices_train_list = []
    matrices_val_list = []
    for train_index, val_index in indices_list:
        x_train_fold, x_test_fold = x_train.iloc[train_index], x_train.iloc[val_index]

        # l_name_train_books is a list of "name_train_books" lists, each such sub-list corresponds to a given century
        l_name_train_books = []
        centuries = [(1400, 1500), (1500, 1600), (1600, 1700), (1700, 1800), (1800, 1900), (1900, 1940), (1940, 2020)]
        for a,b in centuries:
            name_train_books_cent = []
            x_train_fold_cent = x_train_fold[(train_books['Year'] >= a) & (train_books['Year'] < b)]
            for name_book in x_train_fold_cent['Filename']:
                name_train_books_cent.append(name_book[2:-1])
            l_name_train_books.append(name_train_books_cent)

        vocabulary = set()

        # l_name_train_books is a list of "name_train_books" lists, each such sub-list corresponds to a given century
        for i in range(len(l_name_train_books)):
            # build a dictionary by union of dictionaries (one for each century), excluding their intersection
            tf_vectorizer, train_tf = create_vectorizer_tf(l_name_train_books[i], in_stop_words=in_stop_words, max_features=max_features,
                                                          in_max_df=in_max_df, in_min_df=in_min_df)
            #  print("FOLD ", k, "TOP WORDS:\n", top_words_set)
            top_words_set = list(tf_vectorizer.vocabulary_.keys())
            vocabulary = vocabulary.symmetric_difference(top_words_set)

        print(len(vocabulary))

        # get full train and val (not by century)
        name_train_books = []
        name_test_books = []
        for name_book in x_train_fold['Filename']:
            name_train_books.append(name_book[2:-1])

        for name_book in x_test_fold['Filename']:
            name_test_books.append(name_book[2:-1])

        tf_vectorizer, trained_matrix = create_vectorizer_tfidf(name_train_books, vocab=list(vocabulary))
        # get feature vectors of the test set
        list_texts = []
        i = 0
        for file in name_test_books:
            file_opened = open(parts_dir_path + "\\" + file, "r", encoding="ISO-8859-1")
            file_opened = file_opened.read()
            list_texts.append(file_opened)
            i += 1
        val_matrix = tf_vectorizer.transform(list_texts)
        matrices_train_list.append(trained_matrix)
        matrices_val_list.append(val_matrix)
    return matrices_train_list, matrices_val_list



# pre-train vectorizer and tfidf matrix for each fold
def get_tf_tfidf(indices_list, MAX_DF = 1.0, TOP_N_WORDS=30000, MIN_DF = 0.0, STOP_WORDS=None, TFIDF=False,
                            VOC_IDF=False, use_max_features=True, norm='l1', ngram=(1,1), sublinear_tf=False):
    matrices_train_list = []
    matrices_val_list = []
    for train_indexes, val_indexes in indices_list:
        name_train_books = []
        for index in train_indexes:
            name_train_books.append(train_books['Filename'][index][2:-1])

        name_val_books = []
        for index in val_indexes:
            name_val_books.append(train_books['Filename'][index][2:-1])
        train_matrix, val_matrix = find_feature_vectors(name_train_books, name_val_books, MAX_DF = MAX_DF,
                                                         TOP_N_WORDS=TOP_N_WORDS, MIN_DF = MIN_DF,
                                                         STOP_WORDS=STOP_WORDS, TFIDF=TFIDF, VOC_IDF=VOC_IDF,
                                                         use_max_features=use_max_features, norm=norm, ngram=ngram,
                                                         sublinear_tf=sublinear_tf)

        matrices_train_list.append(train_matrix)
        matrices_val_list.append(val_matrix)
    return matrices_train_list, matrices_val_list


def save_kfold(k):
    # Get folds for cross-validation
    y_train = train_books['Year']

    bins = np.linspace(1400, 2020, 7)
    y_binned = np.digitize(y_train, bins)
    kfold_obj = StratifiedKFold(n_splits=k, random_state=None, shuffle=False)
    indices_5fold = list(kfold_obj.split(np.zeros(len(y_binned)), y_binned))

    filename = "indices_5fold"
    # open the file for writing
    fileObject = open(filename, 'wb')
    # this writes the object a to the file
    pickle.dump(indices_5fold, fileObject)
    fileObject.close()

# save list of folds for CV and corresponding tfidf matrices
def save_matrices():
    fileObject = open("indices_5fold", 'rb')
    indices_list = pickle.load(fileObject)
    fileObject.close()

    # Example:
    matrices_train_list, matrices_val_list = get_tf_tfidf(indices_list, MAX_DF = 1.0, TOP_N_WORDS=10000, MIN_DF = 0.0, STOP_WORDS=None,
                                                          TFIDF=True, VOC_IDF=False, use_max_features=True, norm='l1')
    filename = "example_train"
    # open the file for writing
    fileObject = open(filename, 'wb')
    # this writes the object a to the file
    pickle.dump(matrices_train_list, fileObject)
    fileObject.close()
    filename = "example_val"
    # open the file for writing
    fileObject = open(filename, 'wb')
    # this writes the object a to the file
    pickle.dump(matrices_val_list, fileObject)
    fileObject.close()

#save_matrices()
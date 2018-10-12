from sklearn.tree import DecisionTreeRegressor
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tf_tfidf_vectorizer import create_vectorizer_tf, parts_dir_path, top_mean_feats, tf_tfidf_result, stop_words
from tf_tfidf_vectorizer import create_vectorizer_tfidf
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud

model_dir_path = os.path.dirname(os.path.realpath(__file__))
train_books = pd.read_csv(model_dir_path + "\\..\\train.csv")

# print wordclouds for some intuitions
def print_wordclout():
    temp = train_books[(train_books['Year'] >= 1910) & (train_books['Year'] < 1920)]
    y_train = temp['Year']
    x_train = temp.loc[:, temp.columns != 'Year']

    name_train_books = []
    for name_book in x_train['Filename']:
        name_train_books.append(name_book[2:-1])

    text = ""
    for file in name_train_books:
        file_opened = open(parts_dir_path + "\\" + file, "r", encoding="ISO-8859-1")
        file_opened = file_opened.read()
        text += file_opened

    # lower max_font_size
    wordcloud = WordCloud(max_font_size=40, stopwords=None, background_color='white').generate(text)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()


# ================================ building dictionary from words for every century =============================

y_train = train_books['Year']
x_train = train_books.loc[:, train_books.columns != 'Year']

bins = np.linspace(1400, 2020, 7)
y_binned = np.digitize(y_train, bins)

kfold_obj = StratifiedKFold(n_splits=10, random_state=None, shuffle=False)
k = 0
sum_mse = 0
sum_exp_var = 0
sum_mae = 0
sum_r2 = 0
sum_rmse = 0
error_percent_sum = 0

for train_index, test_index in kfold_obj.split(np.zeros(len(y_binned)), y_binned):
    # get train and test filenames
    x_train_fold, x_test_fold = x_train.iloc[train_index], x_train.iloc[test_index]
    y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]

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
        tf_vectorizer, train_tf = create_vectorizer_tf(l_name_train_books[i], in_stop_words=None,
                                                       in_max_df=1.0, in_min_df=0.0, norm=None)
        top_words = top_mean_feats(train_tf, tf_vectorizer.get_feature_names(), top_n=20000)

        top_words_set = set(top_words['feature'])
      #  print("FOLD ", k, "TOP WORDS:\n", top_words_set)
        vocabulary = vocabulary.symmetric_difference(top_words_set)
      #  print(vocabulary)

    print(len(vocabulary))

    # get full train and test (not by century)
    name_train_books = []
    name_test_books = []
    for name_book in x_train_fold['Filename']:
        name_train_books.append(name_book[2:-1])

    for name_book in x_test_fold['Filename']:
        name_test_books.append(name_book[2:-1])

    tf_vectorizer, train_tf = create_vectorizer_tfidf(name_train_books, vocab = list(vocabulary))
    # get feature vectors of the test set
    list_texts = []
    i = 0
    for file in name_test_books:
        # if i >= 10:
        #     break
        # print(i, ": ", file)
        file_opened = open(parts_dir_path + "\\" + file, "r", encoding="ISO-8859-1")
        file_opened = file_opened.read()
        list_texts.append(file_opened)
        i += 1
    test_tf = tf_vectorizer.transform(list_texts)
    #print("test: ", test_tf.shape)
    # declare regressor with L1 regularization TODO: L1 ('mae') runs very slowly, for now I kept L2 ('mse')...
    regr = DecisionTreeRegressor()
    regr = regr.fit(train_tf, y_train_fold)
    y_pred = regr.predict(test_tf)

    # save results of each fold
    d = {'Year' : y_test_fold, 'Prediction' : y_pred}
    df = pd.DataFrame(data=d)
    result = pd.concat([df, x_test_fold], axis=1)
    result.to_csv("cv_fold" + str(k) + ".csv", index=False)

    # evaluate performance:
    # explained variance (1 is the best score); r2 (1 is the best score)
    exp_var = explained_variance_score(y_test_fold, y_pred)
    mse = mean_squared_error(y_test_fold, y_pred)
    mae = mean_absolute_error(y_test_fold, y_pred)
    r2 = r2_score(y_test_fold, y_pred)
    rmse = sqrt(mse)
    print("Fold: ", k)
    print("Explained variance: ", exp_var)
    print("Mean square error score: ", mse, "RMSE (root mean sqaure error): ", rmse)
    print("Mean absolute error: ", mae)
    print("R2 score: ", r2, "\n")
    sum_exp_var += exp_var
    sum_rmse += rmse
    sum_mse += mse
    sum_mae += mae
    sum_r2 += r2

    # calculate the percentage of examples in test set for which the error is less than 50 years;
    # note: the percentage should be very high since even giving all the examples the same year around ~1900
    # will be close enough to most samples (since we have much more examples from those years)
    df = df.sort_values(by=['Year'])
    df['Error'] = abs(df['Year'] - df['Prediction'])
    error_count = len(df[df['Error'] < 50])
    error_percentage = error_count / len(df)
    print("Percentage of examples with error less than 50 years: ", error_percentage)
    error_percent_sum += error_percentage
    # plot error for each example (abs distance)
    if k < 1: # plot for the first iterations to see if plots are generally similar
        df.plot(x='Year', y='Error', kind='line')
        plt.xlabel("Year")
        plt.ylabel("Error")
        plt.title("Decision Tree Regression Error Per Example")
        plt.legend()
        plt.show()

    k += 1

print("Avg Explained Variance: ", sum_exp_var / 10)
print("Avg MSE: ", sum_mse/10)
print("Avg MAE: ", sum_mae/10)
print("Avg R2: ", sum_r2/10)
print("Avg RMSE: ", sum_rmse/10)
print("Avg Percentage of examples with error less than 50 years: ", error_percent_sum/10)

import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

stop_words = ENGLISH_STOP_WORDS.union(['gutenberg', 'project'])

model_dir_path = os.path.dirname(os.path.realpath(__file__))
parts_dir_path = model_dir_path + "\..\dataset\parts_of_books"


# From https://buhrmann.github.io/tfidf-analysis.html
def top_tfidf_feats(row, features, top_n=25):
    ''' Get top n tfidf values in row and return them with their corresponding feature names.'''
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats)
    df.columns = ['feature', 'tfidf']
    return df


# From https://buhrmann.github.io/tfidf-analysis.html
def top_feats_in_doc(Xtr, features, row_id, top_n=25):
    ''' Top tfidf features in specific document (matrix row) '''
    row = np.squeeze(Xtr[row_id].toarray())
    return top_tfidf_feats(row, features, top_n)


# From https://buhrmann.github.io/tfidf-analysis.html with some changes
def top_mean_feats(Xtr, features, grp_ids=None, min_tfidf=0.1, top_n=25):
    ''' Return the top n features that on average are most important amongst documents in rows
        indentified by indices in grp_ids. '''
    if grp_ids:
        D = Xtr[grp_ids].toarray()
    #else:
        #D = Xtr.toarray()

    #D[D < min_tfidf] = 0
    #tfidf_means = np.mean(D, axis=0)

    # Works! Better performance (no need to convert to array)
    test_means = Xtr.mean(axis=0)
    test_means = np.squeeze(np.asarray(test_means))

    #return top_tfidf_feats(tfidf_means, features, top_n)
    return top_tfidf_feats(test_means, features, top_n)


# From https://buhrmann.github.io/tfidf-analysis.html
def plot_tfidf_classfeats_h(dfs):
    ''' Plot the data frames returned by the function plot_tfidf_classfeats(). '''
    fig = plt.figure(figsize=(12, 9), facecolor="w")
    x = np.arange(len(dfs[0]))
    for i, df in enumerate(dfs):
        ax = fig.add_subplot(1, len(dfs), i+1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_frame_on(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.set_xlabel("Mean Tf-Idf Score", labelpad=16, fontsize=14)
        #ax.set_title("label = " + str(df.label), fontsize=16)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
        ax.barh(x, df.tfidf, align='center', color='#3F5D7D')
        ax.set_yticks(x)
        ax.set_ylim([-1, x[-1]+1])
        yticks = ax.set_yticklabels(df.feature)
        plt.subplots_adjust(bottom=0.09, right=0.97, left=0.15, top=0.95, wspace=0.52)
    plt.show()


def top_idf_features(tfidf_vectorizer, top_n=25):
    indices = np.argsort(tfidf_vectorizer.idf_)[::-1]
    features = tfidf_vectorizer.get_feature_names()
    return [features[i] for i in indices[:top_n]]


def train_vectorizer(list_files, vectorizer):
    # Create list with the texts of all parts
    list_texts = []
    i = 0
    for file in list_files:
        file_opened = open(parts_dir_path + "\\" + file, "r", encoding="ISO-8859-1")
        file_opened = file_opened.read()
        list_texts.append(file_opened)
        i += 1

    # Tokenize and build dictionarry
    trained_vectorizer = vectorizer.fit_transform(list_texts)
    return trained_vectorizer


# Create vectorizer using TF-IDF
def create_vectorizer_tfidf(list_files, vocab = None, norm='l1', in_stop_words=ENGLISH_STOP_WORDS, in_max_df=1.0,
                            in_min_df=0.0, max_features=None, ngram=(1,1), sublinear_tf=False):
    # Create a transform
    tfidf_vectorizer = TfidfVectorizer(norm=norm, vocabulary = vocab, use_idf=True, analyzer='word', max_features=max_features,
                                       stop_words=in_stop_words, max_df=in_max_df, min_df=in_min_df, ngram_range=ngram,
                                       sublinear_tf=sublinear_tf)

    return tfidf_vectorizer, train_vectorizer(list_files, tfidf_vectorizer)


# Create vectorizer using TF
def create_vectorizer_tf(list_files, max_features=None, vocab = None, norm='l1',
                         in_stop_words=ENGLISH_STOP_WORDS, in_max_df=1.0, in_min_df=0.0, ngram=(1,1),
                         sublinear_tf=True):
    # Create a transform
    tf_vectorizer = TfidfVectorizer(norm=norm, vocabulary = vocab, max_features = max_features, use_idf=False,
                                    analyzer='word', stop_words=in_stop_words, max_df=in_max_df, min_df=in_min_df,
                                    ngram_range=ngram, sublinear_tf=sublinear_tf)

    return tf_vectorizer, train_vectorizer(list_files, tf_vectorizer)


# Get the top n results according to TF/TF-IDF average of the tokens
def tf_tfidf_result(tf_tfidf_vectorizer, train_tf_tfidf, in_top_n=30000, out_file=False):
    top_feats_tfidf = top_mean_feats(train_tf_tfidf, tf_tfidf_vectorizer.get_feature_names(), top_n=in_top_n)
    #plot_tfidf_classfeats_h([top_feats_tfidf])

    if not out_file:
        return top_feats_tfidf

    # Create & Write top_feats_tfidf (top of the means of the tfidf of each word)
    if not os.path.isfile(model_dir_path + "\\top_feats_tf_tfidf.csv"):
        top_feats_tfidf.to_csv(model_dir_path + "\\top_feats_tf_tfidf.csv")

    return top_feats_tfidf


# Get the top n results according to IDF average of the tokens
def idf_result(tfidf_vectorizer, in_top_n=30000, out_file=False):
    top_feats_idf = top_idf_features(tfidf_vectorizer, top_n=in_top_n)

    if not out_file:
        return top_feats_idf

    # Create & Write top_feats_idf (according to the idf of the words..)
    if not os.path.isfile(model_dir_path + "\\top_feats_idf.txt"):
        with open(model_dir_path + "\\top_feats_idf.txt", "w+") as fp:
            fp.write("\n".join(top_feats_idf))

    return top_feats_idf
    
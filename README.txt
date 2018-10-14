README:
You will find here a quick description of all the files/scripts used in our project.


Files:
/dataset/
    books/
    This directory contains all the books of our dataset.
    Each book has his own .txt file.

    parts_of_books:
    This directory contains the parts of the books that will be used in our experiments.
    All the parts of a same books are kept in a same .txt file.

    {data, test, train, val}.csv:
    The file data.csv contains information of each books used in our dataset :
    - name of the book
    - year retrieved by parsing or scraping
    The files 'train.csv', 'val.csv' and 'test.csv' contains the splits of the dataset
    that will be used for our experiments.

    indices_5fold:
    This file is a serialized list of size 5 as the number of folds that we used in our experiments.
    For each fold, the list contains indexes which correspond to books in the file train.csv.  

/experiments/
    matrices/
    This directory contains all the features matrices we saved. This way, we don't have to
    generate those again at every run.
    (we don't submit it because it weight a lot)

Scripts:
/dataset/
    utils.py:
    This file contains several functions:
    - 'drop_dup()', 'remove_nans()' are used to clean the data.
    - 'plot_histogram()' plot a histogram of the distribution of the data by year.
    - 'split_data()' splits the data to 'train', 'val' and 'test'.
    - 'save_train_val_test()' saves the split.

    books_to_parts.py:
    The main function of this file is 'books_to_parts()' which iterates through the books,
    and for each takes MAX_PARTS_TO_TAKE parts that each contains PART_SIZE lines,
    then saves the parts in the directory '/parts_of_books'.
    MAX_PARTS_TO_TAKE and PART_SIZE are parameters.

    labeling/
        scraping_openlibrary.py:
        The main function of this file is 'scrape_date()' which receives the name of the book
        and the name of the author. It then scrape the website www.openlibrary.org using the
        package 'BeautifulSoup' and extract, if found, the year of the book.

        label_preparation.py:
        The main function of this file is 'parse_files()' which parse every book in our dataset
        to tag them with the year in which they were written.
        We only used books from the Gutenberg Project that most have at the beginning 
        a header containing informations about the books.
        We try in the function 'parse_files()' to recognize the pattern of the header to extract
        the year of the book.

/experiments/
    tf_tfidf_vectorizer.py:
    This file is essentially a wrapper for creating a TfIdfVectorizer from the sklearn library
    with specific parameters and train it with a list of books to get the feature matrix and
    build a dictionnary.
    'create_vectorizer_tf()' returns a TfIdfVectorizer trained and a feature matrix with TF
    values while 'create_vectorizer_tfidf()' returns a feature matrix with TFIDF values.
    There also are functions to get from the vectorizer and the feature matrix the top words
    to build a dictionnary of the top words.
    'tf_tfidf_result()' returns the top words by TF or TFIDF according to the vectorizer
                        received by parameters.
    'idf_result()' returns the top words by IDF.
    The functions that calculate the top words by TF/TF-IDF average were found here:
    https://buhrmann.github.io/tfidf-analysis.html

    save_structs.py:
    This file contains functions that save structures or objects that will be used in our experiments.
    The function 'save_kfold()' uses StratifiedKFold from sklearn to split the train set to k folds
    according to the distribution of the data in the train set, that way each fold keeps the same
    distribution of the data. We chose to split it into 5 folds, this function saves then the folds
    into a file that we called 'indices_5fold'.
    The function 'find_feature_vectors()' returns the feature matrix of a training set and a test set.
    In order to not calculate those matrices everytime, the function 'save_matrices()' saves those
    matrices into a file.
    The function 'get_tf_tfidf()' does the same as 'find_feature_vectors()' but for multiple folds.
    The function 'get_tfidf_manual_dict()' returns the feature matrices for the test set and for each
    fold of the train set but this time with a dictionnary that we built manually according to the
    periods to which the books belong.

    run_models.py:
    This file is a wrapper for different models from sklearn.
    It contains functions to initialize the features matrices of the train and test set, and the target
    values of the train set.
    There are functions to then run a model on the data we initialized with differents parameters
    according to the model.
    The models supported are: 'KNeighborsRegressor', 'DecisionTreeRegressor', 'RandomForestRegressor',
    'Lasso'.
    Resampling is supported using the implementation of 'resampling.py' in the function 'run_resampling()'
    which resamples the data that we initialized.
    Dimensionality reduction is also supported using TruncatedSVD from sklearn in the function
    'run_svd()' which reduces the dimension of the feature matrices we initialized.
    After having run the Lasso model, we can then save the indexes of the features used by the model in
    a file.
    We can then print different metrics of the results using the function 'print_results()'.

    cross_validation.py:
    This file contains the implementation to run cross validation experiments.
    We initialize the feature matrices for the train and validation set with the function 'initialize()'.
    We must initialize here a list of matrices for each fold of the cross validation.
    We can initialize a list of lasso indexes corresponding to the features that will be used from the
    feature matrices.
    We then can run the 'cross_validation()' function giving a name of the model to run and a list of
    parameters that will be passed to the model.
    This function supports Resampling and Dimensionality reduction too.

    tune_feature_vectors.py:
    We run the 'cross_validation' function from this file with differents parameters to tune the models
    and tune the feature matrices.

    resampling.py:
    This file contains the implementations of the resampling algorithms used in our experiments.
    The algorithms supported are:
    - undersampling and oversampling in the function 'under_over_sampling()'
    - smoteR as found in the following paper: https://pdfs.semanticscholar.org/43cd/a672b9ac0833086e19c90d42c2c0fbc361c6.pdf

    validate_test.py:
    We run in this file the validation and test set. It is different from a cross validation run because
    here we don't have the training set separated in different folds.

    utils.py:
    This file contains the functions that calculate all the metrics used in our project.
    It includes functions to plot graphs, histograms and confusion matrices to help us understand
    results that we got.
    
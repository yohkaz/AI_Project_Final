import pandas as pd
import os
import run_models
from save_structs import find_feature_vectors
import pickle

def initialize(train_file, val_file, test_file=None, save_matrices=False, load_matrices=False, train_mat_filename=None,
               test_mat_filename=None, years_filename=None):
    model_dir_path = os.path.dirname(os.path.realpath(__file__))

    train_books = pd.read_csv(model_dir_path + "\\..\\dataset\\" + train_file)
    val_books = pd.read_csv(model_dir_path + "\\..\\dataset\\" + val_file)

    if test_file is not None:
        run_models.initialize_years_file(model_dir_path + "\\..\\dataset\\" + train_file, model_dir_path + "\\..\\dataset\\" + val_file,
                                         model_dir_path + "\\..\\dataset\\" + test_file)
        test_books = pd.read_csv(model_dir_path + "\\..\\dataset\\" + test_file)
        train_books = pd.concat([train_books, val_books], ignore_index=True)
    else:
        run_models.initialize_years_file(model_dir_path + "\\..\\dataset\\" + train_file, model_dir_path + "\\..\\dataset\\" + val_file)
        test_books = val_books

    train_books = train_books.loc[:, train_books.columns != 'Unnamed: 0']
    test_books = test_books.loc[:, test_books.columns != 'Unnamed: 0']

    name_train_books = []
    for index, rows in train_books.iterrows():
        name_train_books.append(train_books['Filename'][index][2:-1])

    name_val_books = []
    for index, rows in test_books.iterrows():
        name_val_books.append(test_books['Filename'][index][2:-1])

    print(name_val_books)

    print(len(name_train_books), len(name_val_books))

    if not load_matrices:
        train_matrix, val_matrix = find_feature_vectors(name_train_books, name_val_books, MAX_DF=1.0,
                                                        TOP_N_WORDS=10000, MIN_DF=0.0,
                                                         STOP_WORDS=None, TFIDF=False, VOC_IDF=False,
                                                         use_max_features=True, norm='l1', sublinear_tf=True)

    if save_matrices:
        # open the file for writing
        fileObject = open(train_mat_filename, 'wb')
        # this writes the object a to the file
        pickle.dump(train_matrix, fileObject)
        fileObject.close()
        # open the file for writing
        fileObject = open(test_mat_filename, 'wb')
        # this writes the object a to the file
        pickle.dump(val_matrix, fileObject)
        fileObject.close()

    y_train = None

    if load_matrices:
        fileObject = open(train_mat_filename, 'rb')
        train_matrix = pickle.load(fileObject)
        fileObject.close()
        fileObject = open(test_mat_filename, 'rb')
        val_matrix = pickle.load(fileObject)
        fileObject.close()
        if years_filename is not None:
            fileObject = open(years_filename, 'rb')
            y_train = pickle.load(fileObject)
            fileObject.close()

    run_models.initialize_matrices(train_matrix, val_matrix, y_train)


def run_validation_test(train_file, val_file, p, n_neighbors, weights,
                test_file=None, save_matrices_1=False, load_matrices=False, train_mat_filename=None,
                test_mat_filename=None, years_filename=None, n_components_1=None, coef=None, n_components_2=None,
                save_matrices_2=False, train_mat_filename_2=None,
                test_mat_filename_2=None, years_filename_2=None):

    initialize(train_file, val_file, test_file, save_matrices_1, load_matrices, train_mat_filename,
               test_mat_filename, years_filename)

    if n_components_1 is not None:
        train_matrix, val_matrix = run_models.run_svd(n_components_1)
    if coef is not None:
        train_matrix, y_train = run_models.run_resampling(coef)
    if n_components_2 is not None:
        train_matrix, val_matrix = run_models.run_svd(n_components_2)

    if save_matrices_2:
        # open the file for writing
        fileObject = open(train_mat_filename_2, 'wb')
        # this writes the object a to the file
        pickle.dump(train_matrix, fileObject)
        fileObject.close()
        # open the file for writing
        fileObject = open(test_mat_filename_2, 'wb')
        # this writes the object a to the file
        pickle.dump(val_matrix, fileObject)
        fileObject.close()
        # open the file for writing
        fileObject = open(years_filename_2, 'wb')
        # this writes the object a to the file
        pickle.dump(y_train, fileObject)
        fileObject.close()

    # Run KNN example:
    run_models.run_knn(p, n_neighbors, weights)
    exp_var, mse, mae, r2, error_percentage, recall, precision = run_models.print_results(plot=True)

if __name__ == '__main__':
    run_validation_test("train.csv", "val.csv", "test.csv", p=2, n_neighbors=3, weights='distance',
                   n_components_1=1000, coef=5, n_components_2=300)
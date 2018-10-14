import os
import pandas as pd
import pickle
import run_models
import utils
from math import sqrt
from resampling import under_over_sampling, smoter
from sklearn.decomposition import TruncatedSVD

# Initialize matrices/years
list_train_matrix = None
list_val_matrix = None
all_years = None
indices_list = None
list_lasso_indexes = None


def initialize(matrices_train_filename, matrices_val_filename, list_lasso_indexes_filename):
    global list_train_matrix
    global list_val_matrix
    global all_years
    global indices_list
    global list_lasso_indexes

    model_dir_path = os.path.dirname(os.path.realpath(__file__))
    train_books = pd.read_csv(model_dir_path + "\\..\\dataset\\train.csv")

    all_years = train_books['Year']

    # open folds and matrices
    fileObject = open(model_dir_path + "\..\dataset\indices_5fold", 'rb')
    indices_list = pickle.load(fileObject)

    fileObject = open(matrices_train_filename, 'rb')
    list_train_matrix = pickle.load(fileObject)

    fileObject = open(matrices_val_filename, 'rb')
    list_val_matrix = pickle.load(fileObject)

    # To use indexes we got from lasso
    if list_lasso_indexes_filename is not None:
        fileObject = open(list_lasso_indexes_filename, 'rb')
        list_lasso_indexes = pickle.load(fileObject)


# Parameters:
# name_model: the options are 'knn', 'decision_tree', 'random_forest', 'lasso'
# list_dict_parameters : a list of dictionaries,
#                        each dictionary contains the parameters to run
# name_param_to_plot : plot a graph for the specific parameter with this name,
#                      if None, don't plot the graph
# plot: plot the histogram/confusion-matrix for each fold and print result
def cross_validation(name_model, list_dict_parameters, name_param_to_plot=None, plot=False, undersample=False,
                     oversample=False, save_after_smote=False, load_after_smote=False):
    global list_train_matrix
    global list_val_matrix
    global all_years
    global indices_list
    global list_lasso_indexes

    print("Running", name_model, "cross-validation:")
    # save the different evaluations of each parameter in an array
    exp_var_res = []
    rmse_res = []
    mae_res = []
    percent_res = []
    precision_res = []
    recall_res = []
    param_list = []
    name_plot = ""
    if load_after_smote:
        fileObject = open("train_after_svd_smote", 'rb')
        train_after_smote = pickle.load(fileObject)
        fileObject.close()
        fileObject = open("years_after_svd_smote", 'rb')
        years_after_smote = pickle.load(fileObject)
        fileObject.close()
        fileObject = open("val_after_svd_smote", 'rb')
        val_after_svd = pickle.load(fileObject)
        fileObject.close()

    for dict_params in list_dict_parameters:
        if save_after_smote:
            train_after_smote = []
            years_after_smote = []

            val_after_svd = []
        k = 0
        sum_mse = 0
        sum_rmse = 0
        sum_exp_var = 0
        sum_mae = 0
        sum_r2 = 0
        sum_error_percentage = 0
        sum_recall = 0
        sum_precision = 0
        print("Parameters values: ", dict_params)
        save_lasso_indexes = []
        filename_lasso_indexes = None
        if name_model == 'lasso' and 'filename' in dict_params:
            filename_lasso_indexes = dict_params['filename']
            dict_params['filename'] = None

        if name_param_to_plot is not None:
            param_list.append(dict_params[name_param_to_plot])

        for i, (train_indexes, val_indexes) in enumerate(indices_list):
            print("Fold:", k)
            year_train_books, year_val_books = all_years.iloc[train_indexes], all_years.iloc[val_indexes]

            train_matrix = list_train_matrix[i]
            val_matrix = list_val_matrix[i]

            if 'n_components' in dict_params:
                # use PCA
                t_svd = TruncatedSVD(n_components=dict_params['n_components'])
                train_matrix = t_svd.fit_transform(train_matrix)
                val_matrix = t_svd.transform(val_matrix)

            if 'coef' in dict_params:
                year_train_books = year_train_books.reset_index(drop=True)
                if undersample:
                    year_train_books, train_matrix = under_over_sampling(train_matrix, year_train_books, dict_params['coef'],
                                                                         weighted=False, undersample=False)
                elif oversample:
                    year_train_books, train_matrix = under_over_sampling(train_matrix, year_train_books, dict_params['coef'],
                                                                         weighted=False, undersample=False)
                else: # smote
                    year_train_books, train_matrix = smoter(train_matrix, year_train_books, dict_params['coef'], 5)

            # use pre-trained smote set
            if load_after_smote:
                train_matrix = train_after_smote[i]
                year_train_books = years_after_smote[i]
                val_matrix = val_after_svd[i]

                train_matrix = train_matrix.todense()
               # val_matrix = val_matrix.todense()

            if list_lasso_indexes is not None:
                # To use indexes we got from lasso
                train_matrix = train_matrix[:, list_lasso_indexes[i]]
                val_matrix = val_matrix[:, list_lasso_indexes[i]]

            if 'n_components_2' in dict_params:
                # use PCA
                t_svd = TruncatedSVD(n_components=dict_params['n_components_2'])
                train_matrix = t_svd.fit_transform(train_matrix)
                val_matrix = t_svd.transform(val_matrix)

            if save_after_smote:
                train_after_smote.append(train_matrix)
                years_after_smote.append(year_train_books)
                val_after_svd.append(val_matrix)

            run_models.initialize_matrices(train_matrix, val_matrix)
            run_models.initialize_years(year_train_books, year_val_books)

            coef = dict_params.pop('coef', None)
            n_components = dict_params.pop('n_components', None)
            n_components_2 = dict_params.pop('n_components_2', None)
            if name_model == 'knn':
                name_plot = "KNeighbors Regressor Error Evaluation"
                run_models.run_knn(**dict_params)
            elif name_model == 'decision_tree':
                name_plot = "Decision Tree Regressor Error Evaluation"
                run_models.run_decision_tree(**dict_params)
            elif name_model == 'random_forest':
                name_plot = "Random Forest Regressor Error Evaluation"
                run_models.run_random_forest(**dict_params)
            elif name_model == 'lasso':
                name_plot = "Lasso Error Evaluation"
                lasso_indexes = run_models.run_lasso(**dict_params)
                save_lasso_indexes.append(lasso_indexes)

            if coef is not None:
                dict_params['coef'] = coef

            if n_components is not None:
                dict_params['n_components'] = n_components

            if n_components_2 is not None:
                dict_params['n_components_2'] = n_components_2

            exp_var, mse, mae, r2, error_percentage, recall, precision = run_models.print_results(plot=plot)

            sum_exp_var += exp_var
            sum_mse += mse
            sum_rmse += sqrt(mse)
            sum_mae += mae
            sum_r2 += r2
            sum_error_percentage += error_percentage
            sum_recall += recall
            sum_precision += precision
            k += 1

        print("Avg Explained Variance: ", sum_exp_var / 5)
        print("Avg MSE: ", sum_mse / 5)
        print("Avg RMSE: ", sum_rmse / 5)
        print("Avg MAE: ", sum_mae / 5)
        print("Avg R2: ", sum_r2 / 5)
        print("Avg percentage of examples with small* error: ", sum_error_percentage / 5)
        print("Avg Precision: ", sum_precision / 5)
        print("Avg Recall: ", sum_recall / 5)

        rmse_res.append(sum_rmse / 5)
        mae_res.append(sum_mae / 5)
        percent_res.append(sum_error_percentage * 100 / 5)
        precision_res.append(sum_precision * 100 / 5)
        recall_res.append(sum_recall * 100 / 5)
        if filename_lasso_indexes is not None:
            fileObject = open(filename_lasso_indexes, 'wb')
            pickle.dump(save_lasso_indexes, fileObject)

        if save_after_smote:
            filename = "train_svd_smote_svd"
            # open the file for writing
            fileObject = open(filename, 'wb')
            # this writes the object a to the file
            pickle.dump(train_after_smote, fileObject)
            fileObject.close()

            filename = "years_svd_smote_svd"
            # open the file for writing
            fileObject = open(filename, 'wb')
            # this writes the object a to the file
            pickle.dump(years_after_smote, fileObject)
            fileObject.close()

            filename = "val_svd_smote_svd"
            # open the file for writing
            fileObject = open(filename, 'wb')
            # this writes the object a to the file
            pickle.dump(val_after_svd, fileObject)
            fileObject.close()

    if name_param_to_plot:
        utils.plot_metrics(name_param_to_plot, param_list, rmse_res, mae_res, percent_res, precision_res, recall_res, name_plot)


def run_cross_validation():
    # Initialize with tfidf directorty for example
    initialize("matrices\\1\\1_train", "matrices\\1\\1_val", None)
    print("Initialization done")

    # Run knn for example:
    list_params_knn = [
        {'p': 2, 'n_neighbors': 5, 'weights': 'uniform'},
    ]
    cross_validation('knn', list_params_knn, 'n_neighbors', plot=False)

    list_params_knn = [
        {'p': 1, 'n_neighbors': 5, 'weights': 'distance'},
        {'p': 1, 'n_neighbors': 7, 'weights': 'distance'}
    ]
    #cross_validation('knn', list_params_knn, 'n_neighbors', plot=True)

    # Run lasso for example:
    list_params_lasso = [
        {'alpha': 0.0001, 'fit_intercept': True, 'tol': 0.01}
    ]
    #cross_validation('lasso', list_params_lasso, plot=False)



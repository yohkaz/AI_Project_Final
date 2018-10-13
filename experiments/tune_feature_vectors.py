import cross_validation

# cross validation for evaluating different feature vectors
def cv_feature_vectors():
    # Initialize with tfidf directorty for example
    cross_validation.initialize("matrices\\20\\20_train", "matrices\\20\\20_val", None)
    print("Initialization done")

    # run cross validation for building feature vectors (knn with default parameters)
    list_params_knn = [
         # {'p': 2, 'n_neighbors': 5, 'weights': 'uniform'}
        # {'p': 2, 'n_neighbors': 5, 'weights': 'uniform', 'coef': 0.5},
        # {'p': 2, 'n_neighbors': 5, 'weights': 'uniform', 'coef': 0.6},
        # {'p': 2, 'n_neighbors': 5, 'weights': 'uniform', 'coef': 0.7},
        # {'p': 2, 'n_neighbors': 5, 'weights': 'uniform', 'coef': 0.8}
        # {'p': 2, 'n_neighbors': 5, 'weights': 'uniform', 'coef': 0.9},
        # {'p': 2, 'n_neighbors': 5, 'weights': 'uniform', 'coef': 1.0}
        # {'p': 2, 'n_neighbors': 5, 'weights': 'uniform', 'coef': 1.0}
        # # {'p': 2, 'n_neighbors': 5, 'weights': 'distance'}
        {'p': 1.5, 'n_neighbors': 1, 'weights': 'distance'},
        {'p': 1.5, 'n_neighbors': 2, 'weights': 'distance'},
        {'p': 1.5, 'n_neighbors': 3, 'weights': 'distance'},
        {'p': 1.5, 'n_neighbors': 4, 'weights': 'distance'},
        {'p': 1.5, 'n_neighbors': 5, 'weights': 'distance'},
        {'p': 1.5, 'n_neighbors': 6, 'weights': 'distance'},
        {'p': 1.5, 'n_neighbors': 7, 'weights': 'distance'},
        {'p': 1.5, 'n_neighbors': 8, 'weights': 'distance'},
        {'p': 1.5, 'n_neighbors': 9, 'weights': 'distance'},
        {'p': 1.5, 'n_neighbors': 10, 'weights': 'distance'},
        {'p': 1.5, 'n_neighbors': 15, 'weights': 'distance'},
        {'p': 1.5, 'n_neighbors': 20, 'weights': 'distance'},
        {'p': 1.5, 'n_neighbors': 25, 'weights': 'distance'}
        # {'p': 1, 'n_neighbors': 5, 'weights': 'uniform', 'n_components_2': 100},
        #  {'p': 1, 'n_neighbors': 5, 'weights': 'uniform', 'n_components_2': 200},
        # # {'p': 1, 'n_neighbors': 5, 'weights': 'uniform', 'n_components': 1000},
        # # {'p': 2, 'n_neighbors': 5, 'weights': 'uniform'}
        # {'p': 1, 'n_neighbors': 5, 'weights': 'uniform', 'n_components_2': 300},
        # {'p': 1, 'n_neighbors': 5, 'weights': 'uniform', 'n_components_2' : 400},
        # {'p': 1, 'n_neighbors': 5, 'weights': 'uniform', 'n_components_2': 500},
        # {'p': 1, 'n_neighbors': 5, 'weights': 'uniform', 'n_components_2': 800},
        # {'p': 1, 'n_neighbors': 5, 'weights': 'uniform', 'n_components_2': 1000}
        # {'p': 2, 'n_neighbors': 5, 'weights': 'uniform', 'coef': 6},
        # {'p': 2, 'n_neighbors': 5, 'weights': 'uniform', 'coef': 7},
        # {'p': 2, 'n_neighbors': 5, 'weights': 'uniform', 'coef': 8}
    ]

    list_params_lasso = [
        # {'alpha': 0.000001},
        # {'alpha': 0.000005},
        # {'alpha': 0.00001},
        # {'alpha': 0.00005, 'filename': "lasso_20_4"}
        # {'alpha': 0.0001},
        # {'alpha': 0.0005},
        # {'alpha': 0.001}
        {'alpha': 0.005}

    ]

    list_params_rf = [
        {'max_features':'auto', 'n_estimators':30, 'max_depth':None, 'bootstrap':True}
    ]

   # cross_validation.cross_validation('knn', list_params_knn, plot=False, undersample=True)
   # cross_validation.cross_validation('knn', list_params_knn, name_param_to_plot="Rare to common examples ratio", plot=False, undersample=True)
   #  name_param_to_plot='n_components'
    cross_validation.cross_validation('knn', list_params_knn, plot=False, name_param_to_plot='n_neighbors',
                                      oversample=False, undersample=False, load_after_smote=True)

if __name__ == '__main__':
    cv_feature_vectors()

import cross_validation

# cross validation for evaluating different feature vectors
def cv_feature_vectors():
    # Example:
    # Initialize with tfidf directorty for example
    cross_validation.initialize("matrices\\20\\20_train", "matrices\\20\\20_val", None)
    print("Initialization done")

    # run cross validation for building feature vectors (knn with default parameters)
    list_params_knn = [
        {'p': 2, 'n_neighbors': 3, 'weights': 'uniform'},
        {'p': 2, 'n_neighbors': 5, 'weights': 'uniform'},
        {'p': 2, 'n_neighbors': 7, 'weights': 'uniform'},
    ]

    cross_validation.cross_validation('knn', list_params_knn)

    list_params_lasso = [
       {'alpha': 0.0001}
    ]

   cross_validation.cross_validation('lasso', list_params_lasso)

if __name__ == '__main__':
    cv_feature_vectors()

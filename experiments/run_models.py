# Initialize matrices/years
train_matrix = None
y_train = None
test_matrix = None
y_test = None
y_pred = None


def initialize_matrices(in_train_matrix, in_test_matrix, in_y_train=None):
    global train_matrix
    global test_matrix
    global y_train

    train_matrix = in_train_matrix
    test_matrix = in_test_matrix

    if in_y_train is not None:
        y_train = in_y_train


def initialize_matrices_file(matrices_train_filename, matrices_test_filename):
    import pickle

    fileObject = open(matrices_train_filename, 'rb')
    train_matrix = pickle.load(fileObject)

    fileObject = open(matrices_test_filename, 'rb')
    test_matrix = pickle.load(fileObject)

    initialize_matrices(train_matrix, test_matrix)
    return train_matrix, test_matrix


def initialize_years(in_y_train, in_y_test):
    global y_train
    global y_test

    y_train = in_y_train
    y_test = in_y_test


def initialize_years_file(train_path, val_path, test_path=None):
    import pandas as pd

    train_books = pd.read_csv(train_path)
    val_books = pd.read_csv(val_path)

    if test_path is not None:
        test_books = pd.read_csv(test_path)
        train_books = pd.concat([train_books, val_books], ignore_index=True)
    else:
        test_books = val_books # change name to 'test' for uniformity

    initialize_years(train_books['Year'], test_books['Year'])
    return train_books['Year'], test_books['Year']


# Run the chosen Model
def run_model(MODEL_OBJ, train_matrix, y_train, test_matrix):
    global y_pred

    regr = MODEL_OBJ.fit(train_matrix, y_train)
    y_pred = regr.predict(test_matrix)

    return regr, y_pred

def run_resampling(coef, undersample=False, oversample=False):
    global train_matrix
    global y_train
    from resampling import under_over_sampling, smoter
    y_train = y_train.reset_index(drop=True)
    if undersample:
        y_train, train_matrix = under_over_sampling(train_matrix, y_train, coef,
                                                             weighted=False, undersample=True)
    elif oversample:
        y_train, train_matrix = under_over_sampling(train_matrix, y_train, coef,
                                                             weighted=False, undersample=False)
    else:  # smote
        y_train, train_matrix = smoter(train_matrix, y_train, coef, 5)
    return train_matrix, y_train


def run_svd(n_components):
    global train_matrix
    global test_matrix
    from sklearn.decomposition import TruncatedSVD
    t_svd = TruncatedSVD(n_components=n_components)
    train_matrix = t_svd.fit_transform(train_matrix)
    test_matrix = t_svd.transform(test_matrix)
    return train_matrix, test_matrix

def run_knn(p, n_neighbors, weights):
    from sklearn.neighbors import KNeighborsRegressor
    from scipy.sparse import issparse
    global train_matrix
    global test_matrix
    if issparse(train_matrix) and p == 1.5: # minkowski distance is only available with dense data
        train_matrix = train_matrix.todense()
        test_matrix = test_matrix.todense()
    regr = KNeighborsRegressor(p=p, n_neighbors=n_neighbors, weights=weights, n_jobs=-1)
    run_model(regr, train_matrix, y_train, test_matrix)


def run_decision_tree(max_leaf_nodes, min_samples_split, criterion, max_depth, min_samples_leaf):
    from sklearn.tree import DecisionTreeRegressor
    regr = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, min_samples_split=min_samples_split,
                                 criterion=criterion, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
    run_model(regr, train_matrix, y_train, test_matrix)


def run_random_forest(max_depth, n_estimators, max_features, bootstrap):
    from sklearn.ensemble import RandomForestRegressor
    regr = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators,
                                  max_features=max_features, bootstrap=bootstrap, n_jobs=-1)
    run_model(regr, train_matrix, y_train, test_matrix)


def run_lasso(alpha, fit_intercept=True, tol=0.0001, max_iter=1000, n_top_coeffs=None, filename=None):
    from sklearn.linear_model import Lasso
    import pickle
    regr = Lasso(alpha=alpha, fit_intercept=fit_intercept, tol=tol, max_iter=max_iter)
    regr, _ = run_model(regr, train_matrix, y_train, test_matrix)

    i = 0
    coeffs = []
    indexes = []
    for coeff in regr.coef_.tolist():
        if coeff != 0:
            coeffs.append(coeff)
            indexes.append(i)
        i += 1

    print(len(coeffs), "words used")
    # Sort indexes by the coeffs & take 'n_top_coeffs' best indexes:
    abs_coeffs = map(abs, coeffs)
    indexes_sorted = [x for _, x in sorted(zip(abs_coeffs, indexes))]

    if n_top_coeffs is not None:
        indexes_sorted = indexes_sorted[-n_top_coeffs:]

    if filename is not None:
        fileObject = open(filename, 'wb')
        pickle.dump(indexes_sorted, fileObject)

    return indexes_sorted


# Print Results:
def print_results(plot=False):
    from utils import print_error_hist, calculate_metrics, print_metrics, plot_conf_matrix
    import pandas as pd
    global y_test
    global y_pred
    global y_train

    exp_var, mse, mae, r2, error_percentage, recall, precision = calculate_metrics(y_test, y_pred, y_train, plot)
    if plot:
        print_metrics(exp_var, mse, mae, r2, error_percentage, recall, precision)
    # return result if average need to be calculated
    return exp_var, mse, mae, r2, error_percentage, recall, precision

# Return the predicted years
def results_to_csv(name_of_books, csv_filename):
    import pandas as pd
    global y_pred
    d = {'Name of book': name_of_books, 'Prediction': y_pred}
    df = pd.DataFrame(data=d)
    df.to_csv(csv_filename, encoding='utf-8', index=False)
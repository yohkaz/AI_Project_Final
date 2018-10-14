import matplotlib.pyplot as plt
import re
from math import sqrt
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.neighbors.kde import KernelDensity
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

plt.rcParams['font.size'] = 10
plt.rcParams['legend.fontsize'] = 9

# return pdf for a given point
def calc_pdf(data):
    #just for testing
    #df = pd.read_csv("tfidf_cv_fold0.csv")
    gkde = stats.gaussian_kde(data)
    # plot example
    ind = np.linspace(1400, 2010, 1000)
    plt.plot(ind, gkde.evaluate(ind),'r',label="KDE estimation",color="blue")
    plt.show()
    print(gkde.integrate_box_1d(1900, 1910))
    return gkde


# gets df with PDF column
# threshold is the pdf value under which examples are under 1800
def rec_prec_formula(df):
    threshold = 0.001
    df['Relevance'] = 1 - df['PDF']
    df = df[(df['Relevance'] > 1 - threshold)]
    if len(df) is 0: # for really bad predictions we might not get anything under the threshold
        return 0
    scaler = MinMaxScaler()
    # scale in the range of 0 to 1
    df['Relevance'] = scaler.fit_transform(df[['Relevance']])
    rel_sum = df['Relevance'].sum()
    if rel_sum == 0.0:
        return 0
    # alpha is a boolean function indicating whether our prediction is 'good', we define 'good' as less than
    # 50 years error
    df['alpha'] = df['Error'] < 50
    df['mult'] = df['alpha'] * df['Relevance']
    res = df['mult'].sum() / rel_sum
    return res

# define positives as approximately Year < 1700
def calc_recall(df, gkde):
    df['PDF'] = df['Year'].apply(gkde.evaluate)
    return rec_prec_formula(df)

def calc_precision(df, gkde):
    # difference from recall is that we calculate relevance for the PREDICTED valued
    df['PDF'] = df['Prediction'].apply(gkde.evaluate)
    return rec_prec_formula(df)

# calculate precision and recall as proposed in the following article: http://www.dcc.fc.up.pt/~ltorgo/Papers/tr09.pdf
def calc_precision_recall(df, y_train):
    gkde = stats.gaussian_kde(y_train)
    recall = calc_recall(df, gkde)
    precision = calc_precision(df, gkde)
    return recall, precision

# plot histogram
def plot_histogram(x, y):
    plt.figure()
    plt.title("Percent of Examples with Small* Error, By Periods")
    plt.bar(x, y, width=50, tick_label = ['1400-\n1500', '1500-\n1600', '1600-\n1700', '1700-\n1800',
                                                        '1800-\n1900', '1900-\n1950', '1950-\n2010'])
    plt.ylabel("Percent")
    plt.xlabel("Years")
    plt.show(block=True)


# plot percentage of examples with under 50 years error, per period
def print_error_hist(df):
    periods = [(1400, 1500), (1500, 1600), (1600, 1700), (1700, 1800), (1800, 1900), (1900, 1950), (1950, 2010)]
    error_per_period = []
    for a, b in periods:
        total_period_len = len(df[(df['Year'] >= a) & (df['Year'] < b)])
        if a >= 1800:
            num_small_error = len(df[(df['Year'] >= a) & (df['Year'] < b) & (df['Error'] < 30)])
        else:
            num_small_error = len(df[(df['Year'] >= a) & (df['Year'] < b) & (df['Error'] < 50)])
        percent_small_error = num_small_error / total_period_len
        error_per_period.append(percent_small_error)
    plot_histogram([1400, 1500, 1600, 1700, 1800, 1900, 2000], error_per_period)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True year')
    plt.xlabel('Predicted year')
    plt.show()


# df should have year and prediction columns
def plot_conf_matrix(df):
    bins = np.linspace(1400, 2010, 7, endpoint=False)
    y_binned = np.digitize(df['Year'], bins)
    pred_binned = np.digitize(df['Prediction'], bins)
    df['Error'] = abs(df['Year'] - df['Prediction'])

    # compute confusion matrix
    cnf_matrix = confusion_matrix(y_binned, pred_binned)
    np.set_printoptions(precision=2)

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=["1400-\n1487", "1487-\n1574", "1574-\n1661", "1661-\n1748",
                                               "1748-\n1835", "1835-\n1922", "1922-\n2010"], normalize=True,
                          title='Normalized confusion matrix')


def calculate_metrics(y, y_pred, y_train, plot=False):
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import explained_variance_score
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import r2_score
    # evaluate performance:
    # explained variance (1 is the best score); r2 (1 is the best score)
    exp_var = explained_variance_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    d = {'Year': y, 'Prediction': y_pred}
    df = pd.DataFrame(data=d)
    df['Error'] = abs(df['Year'] - df['Prediction'])
    error_count = len(df[(df['Year'] < 1800) & (df['Error'] < 50)]) + \
                  len(df[(df['Year'] >= 1800) & (df['Error'] < 30)])
    error_percentage = error_count / len(df)
    recall, precision = calc_precision_recall(df, y_train)
    if plot:
        print_error_hist(df)
        plot_conf_matrix(df)
    return exp_var, mse, mae, r2, error_percentage, recall, precision


def print_metrics(exp_var, mse, mae, r2, error_percentage, recall, precision):
    print("Explained variance: ", exp_var)
    print("Mean square error score: ", mse, "RMSE (root mean square error): ", sqrt(mse))
    print("Mean absolute error: ", mae)
    print("R2 score: ", r2)
    print("Percentage of examples with small* error: ", error_percentage)
    print("Precision: ", precision)
    print("Recall: ", recall)


# unit foramt plot for evaluation
def plot_metrics(name_param, param_list, rmse_res, mae_res, percent_res, precision_res, recall_res, title):
    plt.plot(param_list, rmse_res, label='RMSE')
    plt.plot(param_list, mae_res, label='MAE')
    plt.plot(param_list, percent_res, label='Accuracy*')
    plt.plot(param_list, precision_res, label='Precision')
    plt.plot(param_list, recall_res, label='Recall')
    plt.legend(bbox_to_anchor=(0.9, 1), loc=2, borderaxespad=0.)
    plt.xlabel(name_param)
    plt.ylabel("Metric value")
    plt.grid(alpha=0.5)
    plt.title(title)
    plt.show()



# gets list of strings and plots a graph (only right-formatted  string assumed!)
def plot_based_on_strings(name_param, strings, param_list, title, type):
    rmse_res = []
    mae_res = []
    percent_res = []
    precision_res = []
    recall_res = []
    for string in strings:
        len(strings)
        searchObj = re.findall(r':\s(.[-]?\d+\.\d+)', string)
        print(searchObj)
        rmse_res.append(float(searchObj[0]))
        mae_res.append(float(searchObj[1]))
        percent_res.append(float(searchObj[2])*100)
        precision_res.append(float(searchObj[3])*100)
        recall_res.append(float(searchObj[4])*100)
    if type is 'plot':
        plot_metrics(name_param, param_list, rmse_res, mae_res, percent_res, precision_res, recall_res, title)
    if type is 'bar':
        metrics = pd.DataFrame(
            {'RMSE': rmse_res,
             'MAE': mae_res,
             'Accuracy*': percent_res,
             'Precision': precision_res,
             'Recall' : recall_res
             })
        metrics.plot.bar(zorder=3)
        plt.grid(zorder=0, alpha=0.5)
        plt.xlabel(name_param)
        plt.xticks(np.arange(len(param_list)), param_list, rotation='horizontal')
        plt.legend(bbox_to_anchor=(0.9, 1), loc=2, borderaxespad=0.)
        plt.ylabel("Metric value")
        plt.title(title)
        plt.show()
    if type is 'table':
        ax = plt.subplot(111, frame_on=False)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        rmse_res = [str(i)[:7] for i in rmse_res]
        mae_res = [str(i)[:7] for i in mae_res]
        percent_res = [str(i)[:7] for i in percent_res]
        precision_res = [str(i)[:7] for i in precision_res]
        recall_res = [str(i)[:7] for i in recall_res]
        row_labels = ['RMSE', 'MAE', 'Accuracy*', 'Precision', 'Recall']
        col_labels = param_list
        the_table = plt.table(cellText=[rmse_res, mae_res, percent_res, precision_res, recall_res],
                              colWidths=[0.5] * len(col_labels),
                              rowLabels=row_labels, colLabels=col_labels,
                              cellLoc='center', rowLoc='center')

        plt.show()

def print_from_string_example():
    # Example
    strings = []
    param_list = ['uniform', 'distance']
    title = "Uniform vs Distance"
    plot_based_on_strings("", strings, param_list, title, type='table')

# print_from_string_example()
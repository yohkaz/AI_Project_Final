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

# # TODO: just for testing, remove later
# df = pd.read_csv("cv_fold0.csv")
# calc_pdf(df['Year'])
#df['Error'] = abs(df['Year'] - df['Prediction'])
#print(calc_precision_recall((df)))

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
   # plt.xticks(param_list, [10000, 20000, 30000, "None"], rotation='horizontal')
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

            # fig, ax = plt.subplots()
            # # hide axes
            # fig.patch.set_visible(False)
            # ax.axis('off')
            # ax.axis('tight')
            #
            # ax.table(cellText=metrics.values, colLabels=metrics.columns, loc='center')
            #
            # # fig.tight_layout()
            #
            # plt.show()

def print_from_string_example():
    strings = []
    # 11
    # strings.append("Avg MSE:  1666.9918832374642"
    #                " Avg MAE:  23.95271562819967"
    #                " Avg percentage of examples with small* error:  0.759423797348376"
    #                " Avg Precision:  0.7490860712589378"
    #                " Avg Recall:  0.4660919631425921")
    # # 4
    # strings.append("Avg MSE:  1650.0933226239747 "
    #               "Avg MAE:  23.930904368836117"
    #               " Avg percentage of examples with small* error:  0.7570273450821384"
    #               " Avg Precision:  0.7523514990226683"
    #               " Avg Recall:  0.46898947092204385")
    # # 3
    # strings.append("Avg MSE:  1692.9011373453245"
    #                " Avg MAE:  24.30209218438729"
    #                " Avg percentage of examples with small* error:  0.7557627096125042"
    #                " Avg Precision:  0.7397092856798853"
    #                " Avg Recall:  0.45836513802142476")

    # # 8
    # strings.append("Avg MSE:  1900.7052912696392"
    #               " Avg MAE:  26.717681933672104"
    #               " Avg percentage of examples with small* error:  0.7064751707488461"
    #               " Avg Precision:  0.7204831979578181"
    #               " Avg Recall:  0.39722353741799726")

    # # 9
    # strings.append(" Avg MSE:  2039.0837785153994 "
    #                " Avg MAE:  28.376836628083645"
    #                "Avg percentage of examples with small* error:  0.6744975360895664"
    #                " Avg Precision:  0.7329348343115221"
    #                " Avg Recall:  0.3855989712501418")
    #
    # # 8
    # strings.append("Avg MSE:  1917.5827077634076"
    #                " Avg MAE:  27.219346876135525"
    #                " Avg percentage of examples with small* error:  0.6954931061600167"
    #                " Avg Precision:  0.7530201121845945 "
    #                " Avg Recall:  0.405692557369179")

    # # 1
    # strings.append("Avg MSE:  1711.4471266082298 "
    #                "Avg MAE:  24.436361721874167 "
    #                "Avg percentage of examples with small* error:  0.7556214659820559 "
    #                "Avg Precision:  0.7488893387417307 "
    #                "Avg Recall:  0.4591438282181743")

    # # 19
    # strings.append("Avg MSE:  1673.8534271271951"
    #                " Avg MAE:  23.992538096304465"
    #                "Avg percentage of examples with small* error:  0.7584367890534962"
    #                " Avg Precision:  0.7564781033504172"
    #                " Avg Recall:  0.465240409122738")
    #
    # # 18
    # strings.append("Avg MSE:  1657.3864218368985"
    #                " Avg MAE:  23.951020937283612"
    #                " Avg percentage of examples with small* error:  0.7575912217884343"
    #                " Avg Precision:  0.7519628447362561"
    #                " Avg Recall:  0.4684109691280546")
    #
    # # 17
    # strings.append("Avg MSE:  1698.2985864169718"
    #                " Avg MAE:  24.31029853564258"
    #                " Avg percentage of examples with small* error:  0.755059770690081"
    #                " Avg Precision:  0.744443245271805"
    #                "Avg Recall:  0.4601415643968997")
    #
    # # 6
    # strings.append("Avg MSE:  1730.5963105698677"
    #                " Avg MAE:  24.505850685862168"
    #                " Avg percentage of examples with small* error:  0.7540721681133835"
    #                " Avg Precision:  0.7424803556001628"
    #                " Avg Recall:  0.4471309268908145")

    # # 5
    # strings.append("Avg MSE:  2098.1550537895437 "
    #               "Avg MAE:  26.944250663046716 "
    #               "Avg percentage of examples with small* error:  0.7247671248750404 "
    #               "Avg Precision:  0.5507095397223603 "
    #               "Avg Recall:  0.3304115851966322 ")
    # #
    # # 7
    # strings.append("Avg MSE:  1741.6900397438726"
    #                " Avg MAE:  24.55247312292139"
    #                " Avg percentage of examples with small* error:  0.7516766112511719"
    #                " Avg Precision:  0.715157883304786"
    #                " Avg Recall:  0.45155270982416074")

    # # 2
    # strings.append("Avg MSE:  1857.7271949506317 "
    #                 " Avg MAE:  26.408176690482396"
    #                 " Avg percentage of examples with small* error:  0.7130956924696724 "
    #                 "Avg Precision:  0.7421930810020332"
    #                 " Avg Recall:  0.41779466608307747")
    #

    # # 15
    # strings.append("Avg MSE:  1740.7994249276476"
    #                " Avg MAE:  24.619682829511284"
    #                " Avg percentage of examples with small* error:  0.7488610903624415 "
    #                "Avg Precision:  0.7294435353866969"
    #                " Avg Recall:  0.44925346764019647")

    # # 16
    # strings.append("Avg MSE:  1681.450138854268 "
    #                "Avg MAE:  23.998434618905826"
    #                " Avg percentage of examples with small* error:  0.7571703678011743"
    #                " Avg Precision:  0.7460151172725319"
    #                " Avg Recall:  0.455284621152164")
    #
    # # 12
    # strings.append("Avg MSE:  1679.115651418435"
    #                " Avg MAE:  23.979361561118022"
    #                " Avg percentage of examples with small* error:  0.7557632031057704"
    #                " Avg Precision:  0.7497649584611576"
    #                " Avg Recall:  0.4623588765897071")
    #
    # # 13
    # strings.append("Avg MSE:  1665.6034823674531"
    #                " Avg MAE:  24.02206635791338"
    #                " Avg percentage of examples with small* error:  0.7544981774492616"
    #                " Avg Precision:  0.745552511955068"
    #                " Avg Recall:  0.4569925341833299")
    #
    # # 14
    # strings.append("Avg MSE:  1802.5092766159028"
    #                " Avg MAE:  25.056140569704475"
    #                " Avg percentage of examples with small* error:  0.7466064751755759"
    #                " Avg Precision:  0.7181891055386448"
    #                " Avg Recall:  0.43102881739217846")

    # strings.append("Avg MSE:  1666.9918832374642"
    #                " Avg MAE:  23.95271562819967"
    #                " Avg percentage of examples with small* error:  0.759423797348376"
    #                " Avg Precision:  0.7490860712589378"
    #                " Avg Recall:  0.4660919631425921")
    # strings.append("Avg MSE:  -1666.9918832374642"
    #                " Avg MAE:  23.95271562819967"
    #                " Avg percentage of examples with small* error:  -0.759423797348376"
    #                " Avg Precision:  0.7490860712589378"
    #                " Avg Recall:  0.4660919631425921")

    # strings.append("Avg MSE:  " + str(sqrt(1711.4471266082298) - sqrt(1725.9090795949292)) + " "
    #                "Avg MAE:  " + str(24.436361721874167 - 24.161844393251094) + " "
    #                "Avg percentage of examples with small* error:  " + str(0.7556214659820559 - 0.7602651067584492) + " "
    #                "Avg Precision:  " + str(0.7440218886554611 - 0.7236033447931398) + " "
    #                "Avg Recall:  " + str(0.46096618463944034 - 0.4441799709910164))

    # strings.append("Avg MSE:  1285.9143515453404"
    #                " Avg MAE:  20.379685095397466"
    #                " Avg percentage of examples with small* error:  0.8098458316907736"
    #                " Avg Precision:  0.7477804987067006"
    #                " Avg Recall:  0.5407022373245278")
    #
    # strings.append("Avg MSE:  1196.5272357298213"
    #                " MAE:  19.21970468284809"
    #                " Avg percentage of examples with small* error:  0.8225213176215167"
    #                " Avg Precision:  0.7675115345309593"
    #                " Avg Recall:  0.5805141773354663")
    #
    # strings.append("Avg MSE:  1186.7626311550232"
    #                " Avg MAE:  18.841561899999764"
    #                " Avg percentage of examples with small* error:  0.8299894826272043"
    #                " Avg Precision:  0.7824989919534281"
    #                " Avg Recall:  0.6019029747800051")
    #
    # strings.append("Avg MSE:  1213.2489169014902"
    #                " Avg MAE:  18.924720468961944"
    #                " Avg percentage of examples with small* error:  0.827879974549378"
    #                " Avg Precision:  0.7580871632274799"
    #                " Avg Recall:  0.6184695575135899")

    # # 12
    # strings.append("Avg RMSE:  40.970571853709636"
    #                " Avg MAE:  23.979361561118022"
    #                " Avg percentage of examples with small* error:  0.7557632031057704"
    #                " Avg Precision:  0.7457089336820573"
    #                " Avg Recall:  0.4672400608836571")

    # # 12_1
    # strings.append("Avg RMSE:  37.693877937917826"
    #         " Avg MAE:  23.124239081429895"
    #         " Avg percentage of examples with small* error:  0.7595856715867101"
    #         " Avg Precision:  0.6874182006364911"
    #         " Avg Recall:  0.5429622667489622")
    #
    # # 20
    # strings.append("Avg RMSE:  35.423337047053124"
    #                " Avg MAE:  20.706312859112096"
    #                " Avg percentage of examples with small* error:  0.8074595914242982"
    #                " Avg Precision:  0.7567164064957899"
    #                " Avg Recall:  0.5790026741182036")

    # # SVD on 20
    # strings.append("Avg RMSE:  34.19668323677796"
    #                " Avg MAE:  18.698314725525996"
    #                " Avg percentage of examples with small* error:  0.8343497407401486"
    #                " Avg Precision:  0.7626090158415131"
    #                " Avg Recall:  0.6248641702195897")
    #
    # strings.append("Avg RMSE:  32.92033105945701"
    #                " Avg MAE:  18.251082692765063"
    #                " Avg percentage of examples with small* error:  0.8370329418879784"
    #                " Avg Precision:  0.7734645844325633"
    #                " Avg Recall:  0.627591737259726")
    #
    # strings.append("Avg RMSE:  32.99532033581902"
    #                " Avg MAE:  18.27541093729455"
    #                " Avg percentage of examples with small* error:  0.8353454713430519"
    #                " Avg Precision:  0.7669477428146401"
    #                " Avg Recall:  0.6165146646029237")
    #
    # strings.append("Avg RMSE:  33.36928629760007"
    #                " Avg MAE:  18.495616045709216"
    #                " Avg percentage of examples with small* error:  0.8363295077991326"
    #                " Avg Precision:  0.7904464863705689"
    #                " Avg Recall:  0.6203219727234515")
    #
    # strings.append("Avg RMSE:  33.61084582774515"
    #                " Avg MAE:  18.695809043729305"
    #                " Avg percentage of examples with small* error:  0.8321007739772466"
    #                " Avg Precision:  0.7775757392521708"
    #                " Avg Recall:  0.6073500105410872")
    #
    # strings.append("Avg RMSE:  34.73547805073328"
    #                " Avg MAE:  19.675423644954353"
    #                " Avg percentage of examples with small* error:  0.8197103557026149"
    #                " Avg Precision:  0.7775516031786643"
    #                " Avg Recall:  0.5686469444959239")
    #
    # # ~ 2 ~
    # strings.append("Avg RMSE:  35.353033428573134"
    #                " Avg MAE:  22.342224750705924"
    #                " Avg percentage of examples with small* error:  0.7947834137621307"
    #                " Avg Precision:  0.5718158543660015"
    #                " Avg Recall:  0.5071686409794228")

    # # ~ 3 ~
    # strings.append("Avg RMSE:  34.257451010473986 "
    #                " Avg MAE:  19.79347466813987"
    #                " Avg percentage of examples with small* error:  0.8167506247835012"
    #                " Avg Precision:  0.7553750919960527"
    #                " Avg Recall:  0.5667467125683053")
    #
    # # ~ 4 ~
    # strings.append("Avg RMSE:  36.81874908731035"
    #                " Avg MAE:  21.478844485205844"
    #                " Avg percentage of examples with small* error:  0.7985807850895982"
    #                " Avg Precision:  0.8964818528533856"
    #                " Avg Recall:  0.49553403138072244")
    #
    # # ~ 5 ~
    # strings.append("Avg RMSE:  34.26530793634161"
    #                " Avg MAE:  19.75456893657893"
    #                " Avg percentage of examples with small* error:  0.8177362441825622"
    #                " Avg Precision:  0.9480037843422607"
    #                " Avg Recall:  0.5446166408354698")
    #
    # # ~ 6 ~
    # strings.append("Avg RMSE:  34.50943812861296"
    #                " Avg MAE:  20.101454688680157"
    #                " Avg percentage of examples with small* error:  0.8204159691132309"
    #                " Avg Precision:  0.9330375232509767"
    #                " Avg Recall:  0.5421454216394406")

    # # ~ 7 ~
    # strings.append("Avg RMSE:  33.174539368515624"
    #                " Avg MAE:  18.28813073554846"
    #                " Avg percentage of examples with small* error:  0.8350579349653737"
    #                " Avg Precision:  0.7658341135830447"
    #                " Avg Recall:  0.618945941355802")
    # # ~ 8 ~
    # strings.append("Avg RMSE:  35.61377500568928"
    #                " Avg MAE:  18.532312069192205"
    #                " Avg percentage of examples with small* error:  0.8428064087504499"
    #                " Avg Precision:  0.9243475609915268"
    #                " Avg Recall:  0.6211622605028333")
    #
    # # ~ 9 ~
    # strings.append("Avg RMSE:  35.01513896163234"
    #                " Avg MAE:  18.963995828171807"
    #                " Avg percentage of examples with small* error:  0.8318195836087942"
    #                " Avg Precision:  0.9471356657094281"
    #                " Avg Recall:  0.5336716508586581")
    #
    # # ~ 10 ~
    # strings.append("Avg RMSE:  37.56606123630327"
    #                " Avg MAE:  20.852167979445834"
    #                " Avg percentage of examples with small* error:  0.8047745057845171"
    #                " Avg Precision:  0.8975285828736727"
    #                " Avg Recall:  0.7278687618243336")
    #
    # # ~ 11 ~
    # strings.append("Avg RMSE:  34.94618621376255"
    #                " Avg MAE:  19.515248727134427"
    #                " Avg percentage of examples with small* error:  0.8204052640410706"
    #                " Avg Precision:  0.8978324630158874"
    #                " Avg Recall:  0.6907392477205737")

    # # svd + smote p = 1
    # strings.append("Avg RMSE:  37.44012596158555"
    #                " Avg MAE:  20.31591156905693"
    #                " Avg percentage of examples with small* error:  0.807450864922331"
    #                " Avg Precision:  0.8424156510802823"
    #                " Avg Recall:  0.6942505055287435")
    #
    # # svd + smote p=1.5
    # strings.append("Avg RMSE:  37.06891982265097"
    #                " Avg MAE:  20.24748018253638"
    #                " Avg percentage of examples with small* error:  0.8075934907500768"
    #                " Avg Precision:  0.846436121579482"
    #                " Avg Recall:  0.6965898367842758")
    #
    # # svd + smote with p=2
    # strings.append(" Avg RMSE:  38.036189841193824"
    #                " Avg MAE:  20.640329815867776"
    #                " Avg percentage of examples with small* error:  0.8019573969333365"
    #                " Avg Precision:  0.8409218172756712"
    #                " Avg Recall:  0.6958124003096839")

    # # svd + smote with p=1.5, weights=uniform
    # strings.append("Avg RMSE:  37.06891982265097"
    #                " Avg MAE:  20.24748018253638"
    #                " Avg percentage of examples with small* error:  0.8075934907500768"
    #                " Avg Precision:  0.846436121579482"
    #                " Avg Recall:  0.6965898367842758")
    #
    # # svd + smote with p=1.5, weights=distance
    # strings.append("Avg RMSE:  36.95790663911479"
    #                " Avg MAE:  20.064835904933872"
    #                " Avg percentage of examples with small* error:  0.8112560614747452"
    #                " Avg Precision:  0.8491054799588154"
    #                " Avg Recall:  0.6965898367842758")

    # svd + smote + svd p=1.5, uniform
    strings.append(" Avg RMSE:  34.05526323469635"
                   " Avg MAE:  18.448213537203934"
                   " Avg percentage of examples with small* error:  0.8271750503493992"
                   " Avg Precision:  0.9430531970231015"
                   " Avg Recall:  0.7265823325433433")

    # svd + smote + svd p=1.5, distance
    strings.append("Avg RMSE:  33.995757496287446"
                   " Avg MAE:  18.309016263391364"
                   " Avg percentage of examples with small* error:  0.8304130052399776"
                   " Avg Precision:  0.9449244541278045"
                   " Avg Recall:  0.7265823325433433")

    # svd + smote + svd p=1.5, distance


    # , "lasso f.s.\n+knn", "smoteR\n+knn", "lasso f.s.\n+smoteR\n+knn", "smoteR\n+lasso f.s.\n+knn",

    param_list = ['uniform', 'distance']
    #param_list = ["english", "None"]
    #param_list = [5000, 10000, 20000, 30000]
    #param_list = ['l1', 'None', 'l2']
    #param_list = [5000, 10000, 20000, 30000]
    title = "KNN performance as function of p"
    # title = "Performance as function of normalization"
    # title = "Ignoring stop words vs. including them in dictionary"
    # title = "Performance as function of dictionary size"
    # title = "Ignoe stop words vs. use all words"


    plot_based_on_strings("weights", strings, param_list, title, type='table')

# print_from_string_example()
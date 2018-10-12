import os
import pandas as pd
from scipy import stats
import pickle
import numpy as np
import matplotlib.pyplot as plt
import run_models
from sklearn.preprocessing import MinMaxScaler
from math import ceil
from sklearn.neighbors import NearestNeighbors
import random
from scipy.sparse import csr_matrix, vstack, issparse
from sklearn.metrics.pairwise import pairwise_distances

# plot histogram of the number of texts for each year
def plot_histogram(data):
    plt.figure()
    plt.title("Number of Texts by Year")
    plt.hist(data[~np.isnan(data)], range=(1100, 2010), bins='auto')
    plt.ylabel("Number of Books")
    plt.xlabel("Year")
    plt.show(block=True)


# model_dir_path = os.path.dirname(os.path.realpath(__file__))
# train_books = pd.read_csv(model_dir_path + "\\..\\train.csv")

# under sample most common years, where coef is the desired number of examples from the dominant years, per each example
# from the rare years
def under_over_sampling(train_mat, y_train, coef, undersample, weighted=False):
    # plot_histogram(y_train)
    # print("before undersampling: ", len(y_train))
    df = pd.DataFrame(data=y_train)
    gkde = stats.gaussian_kde(y_train)
    threshold = 0.001
    df['PDF'] = y_train.apply(gkde.evaluate)
    if undersample:
        resample_indices = df[df['PDF'] >= threshold].index # examples we need to make less of
        rest_indices = df[df['PDF'] < threshold].index
        n_u = len(rest_indices) * coef # amount that stays is coef * number of 'rare' examples
    else:
        # coef should be between 0 and 1
        resample_indices = df[df['PDF'] < threshold].index # examples we need to make more of
        rest_indices = df[df['PDF'] >= threshold].index
        n_u = ceil(len(rest_indices) * coef)
    if weighted:
        if undersample:
            df = df[df['PDF'] >= threshold]
        else:
            df = df[df['PDF'] < threshold]
        df['Relevance'] = 1 - df['PDF']
        scaler = MinMaxScaler()
        # scale in the range of 0 to 1
        df['Relevance'] = scaler.fit_transform(df[['Relevance']])
        #print(df[['Relevance', 'Year']])
        df['Relevance'] /= df['Relevance'].sum()
        #print("=======================")
        #print(df[['Relevance', 'Year']])
        if undersample:
            resampled_indices = np.random.choice(resample_indices, n_u, replace=False, p=df['Relevance'])
        else:
            resampled_indices = np.random.choice(resample_indices, n_u, replace=True, p=df['Relevance'])
    else:
        if undersample:
            resampled_indices = np.random.choice(resample_indices, n_u, replace=False)
        else:
            resampled_indices = np.random.choice(resample_indices, n_u, replace=True)
    final_indices = np.concatenate([rest_indices, resampled_indices])
    # print("after undersampling: ", len(final_indices))
    y_train_undersampled = y_train[final_indices] #list(itertools.compress(undersampled_indices, y_train))
    #plot_histogram(y_train_undersampled)
    return y_train_undersampled, train_mat[final_indices ,:]


def gen_synth_cases(rel_train_mat, y_train, over_n, k):
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(rel_train_mat)
    distances, indices = nbrs.kneighbors(rel_train_mat)
    y_train_new = pd.Series([])
    # each element of indices is the k nearest neighbours of an example+itself
    new_examples_mat = None
    for i, neighbours in enumerate(indices):
        example = rel_train_mat[i, :]
        neighbours = np.delete(neighbours, 0) # remove the element itself from the neighbours group
        for j in range(over_n):
            rand_neighbour_index = np.random.choice(neighbours)
            rand_neighbour = rel_train_mat[rand_neighbour_index, :]
            #new_example = []
            diff = example - rel_train_mat[rand_neighbour_index, :]
            if len(diff.shape) > 1:
                random_vec_size = diff.shape[1]
            else:
                random_vec_size = diff.shape
            new_example = example - diff * np.random.uniform(low=0.0, high=1.0, size=random_vec_size)
            # new_example = csr_matrix(new_example)
            new_examples_mat = vstack([new_examples_mat, new_example], 'csr')
            if issparse(example):
                new_example_neighbours = vstack([example, rand_neighbour])
                distances_new = pairwise_distances(X=new_example, Y=new_example_neighbours,
                                                   metric='euclidean').flatten()
            else:
                new_example_neighbours = np.vstack((example, rand_neighbour))
                distances_new = pairwise_distances(X=[new_example], Y=new_example_neighbours, metric='euclidean').flatten()
            print("YES")
            dist_1 = distances_new[0] # distance from new point to original example
            dist_2 = distances_new[1] # distance from new point to original example's chosen neighbour
            # weighted multiplication such that bigger distance -> less impact on result
            sum_dist = dist_1 + dist_2
            if sum_dist == 0.0:
                new_example_year = y_train.loc[i]
            else:
                new_example_year = (dist_2 * y_train.loc[i] + dist_1 * y_train.loc[rand_neighbour_index]) / sum_dist
            y_train_new = y_train_new.append(pd.Series([int(new_example_year)]), i)
    return new_examples_mat, y_train_new

# over_n = number of cases to generate for each "rare" case
# k - number of neightbours to use
def smoter(train_mat, y_train, over_n, k):
    df = pd.DataFrame(data=y_train)
    gkde = stats.gaussian_kde(y_train)
    threshold = 0.001
    df['PDF'] = y_train.apply(gkde.evaluate)
    med = y_train.median() # needed in order to distinguish the 2 different extremes
    rare_l = df[(df['PDF'] < threshold) & (df['Year'] <= med)].index # low extremes
    rel_train_mat_l = train_mat[rare_l, :]
    y_train_l = y_train[rare_l]
    y_train_l = y_train_l.reset_index(drop=True)
    mat_with_new_l, y_train_l_new = gen_synth_cases(rel_train_mat_l, y_train_l, over_n, k)
    rare_h = df[(df['PDF'] < threshold) & (df['Year'] > med)].index # high extremes
    rel_train_mat_h = train_mat[rare_h, :]
    y_train_h = y_train[rare_h]
    y_train_h = y_train_h.reset_index(drop=True)
    mat_with_new_h, y_train_h_new = gen_synth_cases(rel_train_mat_h, y_train_h, over_n, k)
    new_matrix = vstack([train_mat, mat_with_new_l, mat_with_new_h], 'csr')
    y_train = y_train.append(y_train_l_new)
    y_train = y_train.append(y_train_h_new)

    return y_train, new_matrix

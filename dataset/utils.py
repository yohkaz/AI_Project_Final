import re
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

# remove files with the same author and book (=same text)
def drop_dup():
    data = pd.read_csv("Data8.csv")
    data = data.drop_duplicates(subset=['Author', 'Book']).reset_index(drop=True)
    del data['Unnamed: 0']
    data.to_csv("Data9.csv")


# remove lines which couldn't be parsed (bad format of text file)
def remove_nans():
    data = pd.read_csv("Data9.csv")
    data = data[pd.notnull(data['Book'])]
    data.to_csv("Data9.csv")


# plot histogram of the number of texts for each year
def plot_histogram(data):
    plt.figure()
    plt.title("Number of Texts by Year")
    plt.hist(data[~np.isnan(data)], range=(1100, 2010), bins='auto')
    plt.ylabel("Number of Books")
    plt.xlabel("Year")
    plt.show(block=True)


def split_data(data):
    # Create the bins: we have 10,144 examples, take 7 bins (one for each century)
    bins = np.linspace(0, 10144, 7)
    y = data['Year']
    y_binned = np.digitize(y, bins)
    # first split to test set and train+val
    x_train_val, x_test, y_train_val, y_test = train_test_split(data.loc[:, data.columns != 'Year'], y,
                                                        train_size=0.85, test_size=0.15,
                                                        shuffle=True, stratify = y_binned)
    # now train+val are of size 8622
    bins = np.linspace(1400, 2020, 7)
    y_binned = np.digitize(y_train_val, bins)
    # now split the train+val group to train and val
    x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val,
                                                      train_size=70/85,
                                                      test_size=15/85,
                                                      shuffle=True, stratify = y_binned)
    return x_train, x_val, x_test, y_train, y_val, y_test


def save_train_val_test():
    all_data = pd.read_csv("data.csv")
    data = all_data[['Author', 'Book', 'Filename', 'Year']]
    x_train, x_val, x_test, y_train, y_val, y_test = split_data(data)
    train = x_train.assign(Year=y_train.values)
    val = x_val.assign(Year=y_val.values)
    test = x_test.assign(Year=y_test.values)
    train.to_csv("train.csv")
    val.to_csv("val.csv")
    test.to_csv("test.csv")


data = pd.read_csv("data.csv")
save_train_val_test()
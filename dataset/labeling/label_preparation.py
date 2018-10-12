import re
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

np.set_printoptions(threshold=np.nan)

data_size = 10144

# start reading at *** START OF THIS PROJECT GUTENBERG EBOOK
# read first 100 lines (that could be some kind of intro containing the year of the book along with useless text
# then take the next 300 lines to the dataset (unless it's the end of the file)
def parse_files():
    directory_in_str = 'books'
    df = pd.DataFrame(columns=['Filename', 'Book', 'Author', 'Year', 'Years (optional)', 'Optional years exist',
                               'Translated'])

    directory = os.fsencode(directory_in_str)

    for file in os.listdir(directory):
        if file.endswith(b'.pdf'): # file names with "-8" are duplicates
            continue;
        with open(os.path.join(directory, file), encoding = "ISO-8859-1") as f:
            read_intro = False
            intro_line = 0
            # getting author and book name require more work with regex, not necessary right now
            find_book_name = True
            book = None
            author = None
            years = list()
            years_tmp = list()
            translated = False
            optional_y = False

            for line in f:
                if (line.startswith("End of Project") or line.startswith("End of the Project")):
                    break;
                # extract book name and author
                if find_book_name:
                    searchObj = re.match(r'Title: (.*)', line)
                    if searchObj:
                        book = searchObj.group(1)
                    searchObj = re.match( r'Author: (.*)', line)
                    if searchObj:
                        author = searchObj.group(1)
                        find_book_name = False
                    searchObj = re.match(r'Language: (.*)', line)
                    if searchObj:
                        lang = searchObj.group(1)
                        if (lang is not 'English' or lang is not ' English' or lang is not 'English '): break;

                if line.startswith("*** START OF THIS PROJECT GUTENBERG EBOOK") \
                        or line.startswith("***START OF THE PROJECT GUTENBERG EBOOK")\
                        or line.startswith("       *       *       *       *       *"):
                    read_intro = True
                    find_book_name = False
                if (read_intro):
                    if ("Translat" in line or "translat" in line):
                        translated = True
                    searchObj = re.findall(r'(\b[1]\d{3}\b)', line)
                    if searchObj:
                        for year in searchObj:
                            if year not in years_tmp:
                                years_tmp.append(year)
                    intro_line += 1
                if intro_line > 100:
                    break;

            first = True
            for year in years_tmp:
                # years.append(year)
                if first:
                    prev_year = year
                    years.append(prev_year)
                    first = False
                delta = int(prev_year) - int(year)
                if delta > 40 or delta < -40:
                    years.append(year)
                prev_year = year

            # maybe better check it for years_tmp (before delta calc.)
            if len(years) > 0:
                year1 = years[0]
            else:
                year1 = None
            if len(years) > 1:
                optional_y = True
            years_str = " ".join(str(x) for x in years)
            d = {'Filename': [file], 'Book': [book], 'Author': [author], 'Year': [year1], 'Years (optional)': [years_str],
                 'Optional years exist': [int(optional_y)],'Translated' : [int(translated)]}
            df2 = pd.DataFrame(data=d)
            df = df.append(df2, ignore_index=True)

            df.to_csv("Data8.csv")
            f.close()


# parse the additional files (different format)
def parse_additional():
    directory_in_str = 'Files\manually added'
    df = pd.DataFrame(columns=['Filename', 'Book', 'Author', 'Year', 'Years (optional)', 'Optional years exist',
                               'Translated'])

    directory = os.fsencode(directory_in_str)
    i=0
    for file in os.listdir(directory):
        print(i)
        i += 1
        if file.endswith(b'.pdf'): # file names with "-8" are duplicates
            continue
        with open(os.path.join(directory, file), encoding = "ISO-8859-1") as f:
            for line in f:
                continue
                searchObj = re.match(r'Year: (.*)', line)
                if searchObj:
                    year = searchObj.group(1)
                if '===================================================================' in line:
                    break
            d = {'Filename': [file], 'Book': None, 'Author': None, 'Year': [year], 'Years (optional)': None,
                 'Optional years exist': None, 'Translated': None}
            df2 = pd.DataFrame(data=d)
            df = df.append(df2, ignore_index=True)
            f.close()
    df.to_csv("Data_add.csv")


# fill missing labels & make corrections based on existing labels of the same author:
#   - calculate the mean of the known labels for each author
#   - apply the mean instead of each label that is far by more than 50 years from the mean
#   - calculate the new mean for each author and fill missing labels with the new mean
def complete_by_author():
    data = pd.read_csv("Data9.csv")
    data['Mean Year'] = data.groupby('Author')['Year'].transform('mean')
    data['Year'] = np.where(abs(data['Mean Year'] - data['Year']) > 50, data['Mean Year'], data['Year'])
    data['Mean Year'] = data.groupby('Author')['Year'].transform('mean')
    data['Year'] = data['Year'].fillna(data['Mean Year'])
    del data['Mean Year']
    data['Year'] = data['Year'].round()
    data = data.dropna(subset=['Year']).reindex()
    data.to_csv("Data11.csv")

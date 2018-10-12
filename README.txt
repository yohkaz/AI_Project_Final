README:
You will find here a description for all the files/scripts we wrote and used for our project.


dataset/utils.py:
    This file contains several functions:
    - 'drop_dup', 'remove_nans' are used to clean the data.
    - 'plot_histogram' plot a histogram of the distribution of the data by year.
    - 'split_data' splits the data to 'train', 'val' and 'test'.
    - 'save_train_val_test' saves the split.

dataset/books_to_parts.py:
    The main function of this file is 'books_to_parts' which iterates through the books,
    and for each takes MAX_PARTS_TO_TAKE parts that each contains PART_SIZE lines,
    then saves the parts in the directory '/parts_of_books'.
    MAX_PARTS_TO_TAKE and PART_SIZE are parameters.

dataset/{data, test, train, val}.csv:
    The file data.csv contains information of each books used in our dataset :
    - name of the book
    - year retrieved by parsing or scraping
    The files 'train.csv', 'val.csv' and 'test.csv' contains the splits of the dataset
    that will be used for our experiments.


dataset/labeling/scraping_openlibrary.py:
    The main function of this file is 'scrape_date' which receives the name of the book
    and the name of the author. It then scrape the website www.openlibrary.org using the
    package 'BeautifulSoup' and extract, if found, the year of the book.

dataset/labeling/label_preparation.py:
    The main function of this file is 'parse_files' which parse every book in our dataset
    to tag them with the year in which they were written.
    We only used books from the Gutenberg Project that most have at the beginning 
    a header containing informations about the books.
    We try in the function "parse_files" to recognize the pattern of the header to extract
    the year of the book.



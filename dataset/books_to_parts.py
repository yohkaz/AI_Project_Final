import os
import random

# Suppose that 'Books_used' is at ..\..\Books_used
dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = os.path.join(dir_path, os.pardir)
dir_path = os.path.join(dir_path, os.pardir)
path_books_used = dir_path + "\Books_used"

# Parameters to change
PART_SIZE = 500 # in lines
MAX_PARTS_TO_TAKE = 5
MIN_LINES_IN_PARAGRAPH = 5

def lines_of_file(file_opened):
    i = 0
    for line in file_opened:
        i += 1
    return i


# Return index of the beginning of the first paragraph
# (suppose that a paragraph has at least MIN_LINES_IN_PARAGRAPH lines in a row not empty)
def index_first_paragraph(file_opened, line_start, line_end):
    index = line_start
    counter = 0

    max_index = line_start
    max_counter = 0
    for line in file_opened[line_start:line_end]:
        if counter == MIN_LINES_IN_PARAGRAPH:
            return index - MIN_LINES_IN_PARAGRAPH
        elif not line.isspace():
            counter += 1
        else:
            if counter > max_counter:
                max_counter = counter
                max_index = index - max_counter
            counter = 0
        index += 1
    return max_index   # in case there is no paragraph of MIN_LINES_IN_PARAGRAPH lines at least


# Return the following tuple:
#  (start,end)
#  - start: index to the start of the text
#  - end  : index to the end of the text
def limits_text_of_book(file_opened):
    n_lines = lines_of_file(file_opened)

    index_end = n_lines
    index = 0
    for line in file_opened:
        if line.startswith("End of Project") or line.startswith("End of the Project")\
                or line.startswith("If you wish to read the entire context of any of these quotations"):
            index_end = index
            break
        index += 1

    if n_lines < PART_SIZE:
        return index_first_paragraph(file_opened, 0, index_end), n_lines - 3

    index_start = index_first_paragraph(file_opened, 0, index_end)
    index = 0
    for line in file_opened:
        if line.startswith("*** START OF THIS PROJECT GUTENBERG EBOOK") \
                or line.startswith("***START OF THE PROJECT GUTENBERG EBOOK"):
            index_start = index_first_paragraph(file_opened, index + 40, index_end)
            if index_start - index_end < 80:
                index_start = index + 40
        elif line.startswith("       *       *       *       *       *") and index < n_lines / 10:
            index_start = index_first_paragraph(file_opened, index, index_end)
        elif line.startswith("===================================================================") and index < n_lines / 10:
            index_start = index_first_paragraph(file_opened, index, index_end)
        index += 1
    return index_start, index_end


def get_parts_of_book(book_name):
    book_opened = open(os.path.join(path_books_used, book_name), encoding = "ISO-8859-1")
    book_opened = book_opened.readlines()
    index_start, index_end = limits_text_of_book(book_opened)

    #print(index_start)
    #print(index_end)

    n_lines_of_text = index_end - index_start
    n_parts = int((n_lines_of_text / PART_SIZE))

    # Return all the text joined
    if n_parts <= MAX_PARTS_TO_TAKE:
        return ''.join(book_opened[index_start:index_end])

    # Return the parts joined
    parts_to_take = random.sample(range(1, n_parts), MAX_PARTS_TO_TAKE)
    parts_to_take.sort()
    #print(parts_to_take)
    joined_parts = ""
    i = 0
    while i < MAX_PARTS_TO_TAKE:
        start_part = index_start + (parts_to_take[i] * PART_SIZE)
        end_part = index_start + ((parts_to_take[i]+1) * PART_SIZE) - 1
        joined_parts += ''.join(book_opened[start_part:end_part])
        i += 1
    return joined_parts

#print(get_parts_of_book("19537.txt"))
#print(get_parts_of_book("keats-ode-495.txt"))
#print(get_parts_of_book("49396-8.txt"))


def books_to_parts():
    #dir_path = os.path.dirname(os.path.realpath(__file__))
    #dir_path = os.path.join(dir_path, os.pardir)
    #dir_path = os.path.join(dir_path, os.pardir) + "\First Model Data\Parts of Books"

    dir_path = os.path.dirname(os.path.realpath(__file__)) + "\Parts of Books"

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    for file in os.listdir(path_books_used):
        if os.path.isfile(dir_path + "\\" + file):
            continue
        parts_file = open(dir_path + "\\" + file, "w+", encoding = "ISO-8859-1")
        parts_file.write(get_parts_of_book(file))
        parts_file.close()

books_to_parts()


import urllib.request
from bs4 import BeautifulSoup
import pandas as pd

def scrape_date(name_book, author):
    # url
    url = "https://openlibrary.org/search?title="
    if type(name_book) is float:
        url = str(int(name_book))
    else:
        for word in name_book.split():
            url += "+" + word
    url += "&has_fulltext=true"

    if(author != None):
        url += "&author="
        if type(author) is float:
            url = str(int(author))
        else:
            for word in author.split():
                url += "+" + word

    req = urllib.request.Request(url, headers={'User-Agent' : "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36"})
    try:
        page_html = urllib.request.urlopen(req)
        soup_object = BeautifulSoup(page_html, 'html.parser')
        date_box = soup_object.find('span', attrs={'class' : 'resultPublisher'})
        #print(date_box)
        info = date_box.text.split()
        date = info[6]
        #print(date)
    except Exception:
        return None
    return date

# Tests..
# print(scrape_date("Dorothy and the Wizard in Oz", None))
# print(scrape_date("The Grain Of Dust A Novel", None))
# print(scrape_date("Little Lord Fauntleroy", None))
# print(scrape_date("How to Tell Stories to Children", None))
# print(scrape_date("The Little Lame Prince", "Miss Mulock--Pseudonym of Maria Dinah Craik"))

df = pd.DataFrame(columns=['Author', 'Book', 'Scraped Year'])
data = pd.read_csv("Data9.csv")
i = 0
for index, row in data.iterrows():
    if pd.isnull(data.iloc[index, 1]):
        a = None
    else:
        a = row['Author']
    b = row['Book']
    print(i)
    date = scrape_date(b, a)
    d = {'Author' : [a], 'Book' : [b], 'Scraped Year' : [date]}
    df2 = pd.DataFrame(data=d)
    df = df.append(df2, ignore_index=True)
    df.to_csv("test2.csv")
    i += 1

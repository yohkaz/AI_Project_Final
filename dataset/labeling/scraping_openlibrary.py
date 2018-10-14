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


import requests
from bs4 import BeautifulSoup
def scrap(url):
    response = requests.get(url)
    text = ""
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        title = soup.find('title')
        text += title.get_text().strip() + "\n"
        paragraphs = soup.find_all('p')
        for idx, para in enumerate(paragraphs, start=1):
            t = para.get_text().strip()
            if t:
                text += f" {t}"
        return text
    else:
        return "Error: Unable to fetch the page."

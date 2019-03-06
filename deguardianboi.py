import requests
from bs4 import BeautifulSoup


class DeGuardianBoi:

    def __init__(self):
        self.corpus = {}

    @staticmethod
    def get(url):
        try:
            return requests.get(url)
        except requests.exceptions.ConnectionError:
            pass

    def _get_article(self, url):
        response = self.get(url)
        return BeautifulSoup(response.content, features="html.parser")

    def fetch_json(self, url):
        response = self.get(url)
        return response.json()

    def fetch_article(self, url):
        soup = self._get_article(url)
        if soup is None:
            return None
        try:
            p_list = soup.find('div', {'itemprop': 'articleBody'}).find_all('p', recursive=False)
            article = [p.text for p in p_list if p.text]
            return article
        except:
            return False


if __name__ == '__main__':
    bb = DeGuardianBoi()
    bb.fetch_article(
        'https://www.theguardian.com/technology/2019/feb/24/are-you-being-scanned-how-facial-recognition-technology-follows-you-even-as-you-shop')

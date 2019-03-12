import nltk
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords


class DeCleanerBoi:
    def __init__(self):
        nltk.download('punkt')
        nltk.download('stopwords')

    @staticmethod
    def clean_article(art):
        tokens = []
        for sent in sent_tokenize(art):
            for word in word_tokenize(sent):
                word = word.lower()
                if len(word) < 2 or word in stopwords.words('english'):
                    continue
                tokens.append(word)
        return tokens


if __name__ == '__main__':
    dcb = DeCleanerBoi()
    w = dcb.clean_article(
        'Lorem ipsum giat pharetra, dui in er accumsan ridiculus, interdum nisl in! Posuere platea gravida ullamcorper augue egestas tristique rutrum montes at quis sociis neque rhoncus eros, et volutpat lectus sodales curae cubilia cras primis per orci praesent penatibus lacinia. Tincidunt tempus porttitor lacus montes iaculis vulputate orci, nam eget porta mattis cubilia cum, imperdiet in dui rutrum facilisis ut. Ante vestibulum blandit vel id natoque aliquet mollis suscipit nostra rhoncus dis, netus mus auctor curabitur donec venenatis at iaculis cubilia posuere ultricies, nascetur lacus malesuada purus quis magnis egestas risus porta non.')
    print(w)

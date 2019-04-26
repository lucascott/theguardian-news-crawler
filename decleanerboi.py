import nltk
import spacy
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords


class DeCleanerBoi:
    def __init__(self):
        nltk.download('punkt')
        nltk.download('stopwords')
        self.stemmer = nltk.SnowballStemmer('english')
        self.nlp = spacy.load("en_core_web_sm")
        self.nlp.remove_pipe('tagger')
        self.nlp.remove_pipe('parser')
        self.nlp.remove_pipe('ner')

    def clean_article(self, art):

        tokens = []
        for sent in sent_tokenize(art):
            for word in word_tokenize(sent):
                word = word.lower()
                tokens = list(map(lambda w: self.stemmer.stem(w), tokens))
                tokens = [w for w in tokens if w.isalpha()]
                if len(word) < 2 or word in stopwords.words('english'):
                    continue
                tokens.append(word)
        return tokens

    def SPACYPipeline(self, doc):
        """
        Transform the corpus with Spacy library
        :param doc:
        :return: list of lists of tokens (strings) e.g. [['hello', ''world'],['how', 'are', 'you']]
        """

        tokens = self.nlp(doc)

        new_tokens = []
        for w in tokens:
            # if it's not a stop word or punctuation mark, add it to our article!
            lemma = w.lemma_  # lemmatized version of the word
            if lemma != '\n' and len(lemma) > 2 and (
                    not w.is_stop or lemma == 'not') and not w.is_punct and not w.like_num:
                new_tokens.append(lemma)

        return new_tokens


if __name__ == '__main__':
    dcb = DeCleanerBoi()
    w = dcb.clean_article(
        'Lorem ipsum giat pharetra, dui in er accumsan ridiculus, interdum nisl in! Posuere platea gravida ullamcorper augue egestas tristique rutrum montes at quis sociis neque rhoncus eros, et volutpat lectus sodales curae cubilia cras primis per orci praesent penatibus lacinia. Tincidunt tempus porttitor lacus montes iaculis vulputate orci, nam eget porta mattis cubilia cum, imperdiet in dui rutrum facilisis ut. Ante vestibulum blandit vel id natoque aliquet mollis suscipit nostra rhoncus dis, netus mus auctor curabitur donec venenatis at iaculis cubilia posuere ultricies, nascetur lacus malesuada purus quis magnis egestas risus porta non.')
    print(w)

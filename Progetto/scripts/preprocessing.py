from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet = True)
nltk.download('wordnet', quiet = True)

class Preprocesser:
    __stopword_list = stopwords.words('english')

    @staticmethod
    def __remove_html_tags(text):
        return BeautifulSoup(text, 'lxml').text

    @staticmethod
    def __remove_special_characters (text):
        pattern=r'[^a-zA-Z\s]'
        return re.sub(pattern,' ',text)

    @staticmethod
    def __stemmer(text):
        ps= PorterStemmer()
        text= ' '.join([ps.stem(word) for word in text.split()])
        return text

    @staticmethod
    def __lemmatize (text):
        lm = WordNetLemmatizer()
        text = ' '.join([lm.lemmatize(word) for word in text.split()])
        text = ' '.join([lm.lemmatize(word, 'v') for word in text.split()]) #verbs
        return text

    @staticmethod
    def __remove_stopwords( text):
        tokens = ToktokTokenizer().tokenize(text)
        tokens = [token.strip() for token in tokens]
        filtered_tokens = [token for token in tokens if token.lower() not in Preprocesser.__stopword_list]
        filtered_text = ' '.join(filtered_tokens)    
        return filtered_text

    @staticmethod
    def text_preprocessing(text):
        text = text.lower()
        text = Preprocesser.__remove_html_tags(text)
        text = Preprocesser.__remove_special_characters (text)
        text = Preprocesser.__remove_stopwords(text)
        text = Preprocesser.__lemmatize(text)
        text = Preprocesser.__stemmer(text)
        return text

    @staticmethod
    def raw_text_preprocessing(text):
        text = text.lower()
        text = Preprocesser.__remove_html_tags(text)
        return text

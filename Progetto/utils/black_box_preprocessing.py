
import pickle
from nltk.tokenize.treebank import TreebankWordTokenizer

class BlackBoxPreprocesser(object):
    
    def __init__(self):
        with open('./pickle_data/preprocesser_utils/utils.pickle', 'rb') as f:
            symbols_to_delete, symbols_to_isolate = pickle.load(f)
            self.__symbols_to_delete = symbols_to_delete
            self.__symbols_to_isolate = symbols_to_isolate
        f.close()
        self.__tree_bank_word_tokenizer = TreebankWordTokenizer()
    
    def __handle_symbols(self, text):
        for symbol in self.__symbols_to_delete:
            text = text.replace(symbol, ' ')
        for symbol in self.__symbols_to_isolate:
            text = text.replace(symbol, ' ' + symbol + ' ')
        return text
    
    def __handle_contractions(self, text):
        text = self.__tree_bank_word_tokenizer.tokenize(text)
        return ' '.join(text)
    
    def __fix_quotes(self, text):
        return ' '.join(w[1:] if w.startswith("'") and len(w) > 1 else w for w in text.split())
        
    def preprocess_text(self, text):
        text = text.lower()
        text = self.__handle_symbols(text)
        text = self.__handle_contractions(text)
        text = self.__fix_quotes(text)
        return text

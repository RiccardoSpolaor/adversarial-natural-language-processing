
import tensorflow as tf
from scripts.preprocessing import Preprocesser
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

class BlackBox:
    
    def __init__(self):
        with open('pickle\\tokenizer.pickle', 'rb') as f:
            tokenizer, maxlen = pickle.load(f)
            self.__tokenizer = tokenizer
            self.__maxlen = maxlen
        self.__model = tf.keras.models.load_model("models\\best_model.h5")
        
    def __text_preprocessing(self, text):
        return Preprocesser.text_preprocessing(text)      
        
    def __tokenize(self, text):
        sequences = self.__tokenizer.texts_to_sequences(text)
        return pad_sequences(sequences, maxlen = self.__maxlen)
        
    def predict_sentiment(self, text):
        text = self.__text_preprocessing(text)
        seq = self.__tokenize([text])
        return self.__model.predict(seq).take(0)
    
#    def evaluate(self, test, label):
#        self.__model.evaluate(test,label)

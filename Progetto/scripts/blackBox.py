
import tensorflow as tf
from scripts.preprocessing import Preprocesser as preprocesser
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

class BlackBox:
    
    def __init__(self):
        with open('pickle\\tokenizer.pickle', 'rb') as f:
            self.__tokenizer, self.__maxlen = pickle.load(f)
        
        self.__model = tf.keras.models.load_model("models\\best_model.h5")
        
    def __text_preprocessing(self, text):
        text = preprocesser.text_preprocessing(text)      
        
    def __tokenize(self, text):
        
        sequences = self.__tokenize.texts_to_sequences(text)

        return pad_sequences(sequences, maxlen = self.__maxlen)
        
    def predict_sentiment(self, text):
        text = self.__text_preprocessing(text)
        text = [text]
        text = self.__tokenize(text)
        return self.__model.predict(text).take(0)
    
    def evaluate(self, test, label):
        self.__model.evaluate(test,label)

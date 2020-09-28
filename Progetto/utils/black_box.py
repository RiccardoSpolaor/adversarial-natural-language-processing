
import tensorflow as tf
import numpy as np
import os
from utils.black_box_preprocessing import BlackBoxPreprocesser
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

class BlackBox(object):
    
    def __init__(self):
        with open(os.path.join('./pickle_data/preprocesser_utils/tokenizer.pickle'), 'rb') as f:
            tokenizer, MAXLEN = pickle.load(f)
            self.__tokenizer = tokenizer
            self.__MAXLEN = MAXLEN
        f.close()
        self.__preprocesser = BlackBoxPreprocesser()
        self.__model = tf.keras.models.load_model(os.path.join('./models/imdb/best_model.h5'))
        
    def __text_preprocessing(self, text):
        return self.__preprocesser.preprocess_text(text)      
        
    def __tokenize(self, text):
        sequences = self.__tokenizer.texts_to_sequences(text)
        return pad_sequences(sequences, maxlen = self.__MAXLEN)
        
    def predict_sentiment(self, text):
        text = self.__text_preprocessing(text)
        seq = self.__tokenize([text])
        return self.__model.predict(seq).take(0)
    
    def evaluate(self, test, label):
        test = [self.__preprocesser.preprocess_text(text) for text in test]
        test_sequences = self.__tokenizer.texts_to_sequences(test)
        test_data = pad_sequences(test_sequences, maxlen = self.__MAXLEN)
        label = np.asarray(label)
        self.__model.evaluate(test_data,label)

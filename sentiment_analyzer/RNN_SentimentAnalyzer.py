import tensorflow as tf
import numpy as np
import pickle
import string
import re
import json

from nltk.tokenize import word_tokenize  # Tokenisasi teks
from nltk.corpus import stopwords  # Daftar kata-kata berhenti dalam teks

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory  # Stemming (penghilangan imbuhan kata) dalam bahasa Indonesia
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory  # Menghapus kata-kata berhenti dalam bahasa Indonesia

import nltk
nltk.download('punkt')  # Mengunduh dataset yang diperlukan untuk tokenisasi teks.
nltk.download('punkt_tab')
nltk.download('stopwords')  # Mengunduh dataset yang berisi daftar kata-kata berhenti (stop words) dalam berbagai bahasa.


def cleaningText(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text) # remove mentions
    text = re.sub(r'#[A-Za-z0-9]+', '', text) # remove hashtag
    text = re.sub(r'RT[\s]', '', text) # remove RT
    text = re.sub(r"http\S+", '', text) # remove link
    text = re.sub(r'[0-9]+', '', text) # remove numbers
    text = re.sub(r'[^\w\s]', '', text) # remove punctuation


    text = text.replace('\n', ' ') # replace new line into space
    text = text.translate(str.maketrans('', '', string.punctuation)) # remove all punctuations
    text = text.strip(' ') # remove characters space from both left and right text
    return text

def casefoldingText(text): # Converting all the characters in a text into lower case
    text = text.lower()
    return text

def tokenizingText(text): # Tokenizing or splitting a string, text into a list of tokens
    text = word_tokenize(text)
    return text

def filteringText(text): # Remove stopwors in a text
    listStopwords = set(stopwords.words('indonesian'))
    listStopwords1 = set(stopwords.words('english'))
    listStopwords.update(listStopwords1)
    listStopwords.update(['iya','yaa','gak','nya','na','sih','ku',"di","ga","ya","gaa","loh","kah","woi","woii","woy"])
    filtered = []
    for txt in text:
        if txt not in listStopwords:
            filtered.append(txt)
    text = filtered
    return text


def fix_slangwords(text):
    words = text.split()
    fixed_words = []

    with open('slangwords.txt', 'r', encoding='utf-8') as file:
        slangwords = json.load(file)

    for word in words:
        if word.lower() in slangwords:
            fixed_words.append(slangwords[word.lower()])
        else:
            fixed_words.append(word)

    fixed_text = ' '.join(fixed_words)
    return fixed_text

def stemmingText(text):
    # Membuat objek stemmer
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    words = text.split()
    stemmed_words = [stemmer.stem(word) for word in words]
    stemmed_text = ' '.join(stemmed_words)

    return stemmed_text

def toSentence(list_words): # Convert list of words into sentence
    sentence = ' '.join(word for word in list_words)
    return sentence


class SentimentAnalyzer:
    def preProcessText(self, text:str):
        text = casefoldingText(text)
        text = fix_slangwords(text)
        text = cleaningText(text)
        text = stemmingText(text)
        text = tokenizingText(text)
        text = filteringText(text)
        text = toSentence(text)
        return text

    def __init__(self):
        with open('model/tokenizer.pickle', 'rb') as handle:
            self.tokenizer = pickle.load(handle)

        with open('model/label_encoder.pickle', 'rb') as handle:
            self.label_encoder = pickle.load(handle)
            
        self.model = tf.keras.models.load_model('model/sentiment_analysis_model_rnn.h5')
        
    def __call__(self, text: str):
        try:
            text = self.preProcessText(text)
            sequences = self.tokenizer.texts_to_sequences([text])
            max_length = 100
            padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_length, padding='post')

            predictions = self.model.predict(padded_sequences)
            result = np.argmax(predictions)
            self.prediction = self.label_encoder.inverse_transform([result])[0]
            return self.prediction
        except Exception as E:
            print(E)
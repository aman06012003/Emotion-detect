import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf 
import nltk
nltk.download('stopwords')
#text preprocessing
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re

def preprocess(line):
    ps = PorterStemmer()

    review = re.sub('[^a-zA-Z]', ' ', line) #leave only characters from a to z
    review = review.lower() #lower the text
    review = review.split() #turn string into list of words
    #apply Stemming 
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')] #delete stop words like I, and ,OR   review = ' '.join(review)
    #trun list into sentences
    return " ".join(review)


st.write("""
# Emotion detection app
This app detects the emotion of a given text""")

st.header('User Input')
def user_input():
    text = st.text_input('Enter your sentence: ')
    # st.write(text)
    encoder = pickle.load(open('encoder.pkl', 'rb'))
    cv = pickle.load(open('CountVectorizer.pkl', 'rb'))
    model=tf.keras.models.load_model('emt_model.h5')
    input=preprocess(text)

    array = cv.transform([input]).toarray()

    pred = model.predict(array)
    a=np.argmax(pred, axis=1)
    prediction = encoder.inverse_transform(a)[0]
    return prediction


string = ''
string = user_input()
if st.button('Predict'):
    st.write(string)
else:
    st.write('')


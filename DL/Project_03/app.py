import numpy as np
import streamlit as st

import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model


word_index = imdb.get_word_index()
reversed_word_index = {value:key for key, value in word_index.items()}
model = load_model('rnn_imdb.h5')


def decode_review(review):
    return ' '.join([reversed_word_index.get(i-3, '?') for i in review])

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2)+3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review
    
def predict_sentiment(review):
    preprocessed_text = preprocess_text(review)
    prediction = model.predict(preprocessed_text)

    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment, prediction[0][0]
    

## Streamlit App
st.title('Movie Review Sentiment Prediction')
st.write('Enter a review!')
review = st.text_input()

# Make prediction
if st.button('Classify'):
    processed_input = preprocess_text(review)
    sentiment, score = predict_sentiment(processed_input)

    # Display the result
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Score: {score}')
else:
    st.write('Please enter a movie review')


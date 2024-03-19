import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


model = load_model('DisasterTweets.h5')


max_w = 10000
tokenizer = Tokenizer(num_words=max_w, oov_token='<OOV>')
tokenizer.fit_on_texts(train_df["text"])


def preprocess_input(phrase):
    seq = tokenizer.texts_to_sequences([phrase])
    padded = pad_sequences(seq, maxlen=200, padding='post', truncating='post')
    return padded
def predict_disaster(phrase):
    input_phrase = preprocess_input(phrase)
    prediction = model.predict(input_phrase
    prediction = 1 if prediction > 0.5 else 0
    return prediction


def run_application():
    while True:
        phrase = input("Enter a phrase: ")
        if phrase.lower() == 'exit':
            print("Exiting...")
            break
        prediction = predict_disaster(phrase)
        if prediction == 1:
            print("The phrase indicates a disaster.")
        else:
            print("The phrase does not indicate a disaster.")

# Run the application
if __name__ == "__main__":
    run_application()
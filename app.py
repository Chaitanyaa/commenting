from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
#from google.cloud import storage #Added

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
import pickle

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# @title Imports

# DataFrame
import pandas as pd
# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer

# Keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Conv1D, MaxPooling1D, LSTM, GRU
from keras import utils
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

# nltk
import nltk
from nltk.corpus import stopwords
from  nltk.stem import SnowballStemmer
#nltk.download('stopwords')
stop_words = stopwords.words("english")
stemmer = SnowballStemmer("english")

# Word2vec
#import gensim

# Utility
import re
import numpy as np
import os
from collections import Counter
import logging
import time
import pickle
import itertools
import h5py


# Set log
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
from tensorflow.keras.models import load_model
MODEL_PATH1 = 'models/cnn_bidirectional_gru_model.h5'
MODEL_PATH2 = 'models/tokenizer_cnn_bidirectional_lstm_model.pkl'
# Load your trained model
sent_model = load_model(MODEL_PATH1)
with open(MODEL_PATH2, 'rb') as handle:
    sent_tokenizer = pickle.load(handle)
    #model._make_predict_function()          # Necessary
print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')

#Added
#MODEL_BUCKET = os.environ['MODEL_BUCKET']
#MODEL_PATH1 = os.environ['MODEL_FILENAME1']
#MODEL_PATH2 = os.environ['MODEL_FILENAME2']

#global MODEL
#client = storage.Client()
#bucket = client.get_bucket(MODEL_BUCKET)
#blob1 = bucket.get_blob(MODEL_PATH1)
#s1 = blob1.download_as_string()
#blob2 = bucket.get_blob(MODEL_PATH2)
#s2 = blob2.download_as_string()
#sent_tokenizer = pickle.load(s2)
#sent_model = load_model(s1)


print('Model loaded. Check http://127.0.0.1:5000/')




from collections import OrderedDict
from textblob import TextBlob

def decode_sentiment(score, include_neutral=True):
    #if include_neutral:        
     #   label = 'NEUTRAL'
      #  if score <= 0.4:
       #     label = 'NEGATIVE'
       # elif score >= 0.5:
        #    label = 'POSITIVE'

        #return label
    #else:
     #   return 'NEGATIVE' if score < 0.4 else 'POSITIVE'
     if include_neutral:        
        label = "NEUTRAL"
        if score <= 0.5:
            label = "NEGATIVE"
        elif score >= 0.6:
            label = "POSITIVE"

        return label
     else:
        return 'NEGATIVE' if score < 0.5 else 'POSITIVE'

      

def predicts(model, tokenizer, text, decode_type, include_neutral=True):
    #start_at = time.time()
    # Tokenize text
    x_test = pad_sequences(tokenizer.texts_to_sequences([text]),maxlen=300)
    print(text)
    # Predict
    score = model.predict([x_test])[0]
    
    if decode_type == 'sent':
      label = decode_sentiment(score, include_neutral=include_neutral)
      #return f"Sentiment: {label}, Score: {float(score)}"
      return f"{label},{float(score)}"
    elif decode_type == 'tox':
      #label = decode_toxicity(score, include_neutral=False)
      return f"Toxicity: {label}, Score: {float(score)}"
    elif decode_type == 'pol':
      #label = decode_pol(score, include_neutral=False)
      return f"Political_Leaning: {label}, Score: {float(score)}"
    


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        comment = request.form['comment']
        print(comment)
        result = predicts(sent_model, sent_tokenizer,comment, decode_type='sent')              # Convert to string
        return result
    return None


if __name__ == '__main__':
    # app.run(port=5002, debug=True)
	#app.run(host="127.0.0.1", port=8080, debug=True) #added
    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()

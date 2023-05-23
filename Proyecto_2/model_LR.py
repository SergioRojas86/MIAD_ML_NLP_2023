import pandas as pd
import numpy as np
import joblib
import sys
import os
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

def predict_genre(features):
    
    regressor = joblib.load(os.path.dirname(__file__) + '/movie_genres_LR.pkl')
    vectorizer =joblib.load(os.path.dirname(__file__) + 'trained/vectorizer.pkl')
    
    #le = MultiLabelBinarizer()
    tfidf_vectorizer = TfidfVectorizer()
    
    def clean_text(text):
        text = re.sub((r'[^\w\s]'),'', text).lower() 
        text = re.sub((r'\d+'),'', text).lower()
        text = re.sub((r'_+'),'', text).lower()
        wnl = nltk.stem.WordNetLemmatizer()
        stopwords = nltk.corpus.stopwords.words('english')
        words = re.sub(r'[^\w\s]', ' ', text).split()
        return ' '.join([wnl.lemmatize(word) for word in words if word not in stopwords])
    
    df5000 = pd.DataFrame(columns=['plot'],index=range(1))
    df5000['plot'] = features
    df5000['clean_plot'] = df5000['plot'].apply(clean_text)
     
    q = df5000['clean_plot']
    q_vec = vectorizer.transform(q)
    
    # Make prediction
    q_pred = regressor.predict(q_vec)

    return le.inverse_transform(q_pred)

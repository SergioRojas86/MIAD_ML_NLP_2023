import pandas as pd
import numpy as np
import joblib
import sys
import os
import re
import nltk

def prediccion_genero(features):
    
    LReg = joblib.load(os.path.dirname(__file__) + '/genero_peliculas.pkl')
    tfidfvectorizer = joblib.load(os.path.dirname(__file__) + '/tfidfvectorizer.pkl')
    mlb =joblib.load(os.path.dirname(__file__) + '/mlb.pkl')
    
    def clean_text(text):
        text = re.sub((r'[^\w\s]'),'', text).lower() 
        text = re.sub((r'\d+'),'', text).lower()
        text = re.sub((r'_+'),'', text).lower()
        wnl = nltk.stem.WordNetLemmatizer()
        stopwords = nltk.corpus.stopwords.words('english')
        words = re.sub(r'[^\w\s]', ' ', text).split()
        return ' '.join([wnl.lemmatize(word) for word in words if word not in stopwords])
    
    df = pd.DataFrame(columns=['plot'],index=range(1))
    df['plot'] = features
    df['clean_plot'] = df['plot'].apply(clean_text)
     
    q = df['clean_plot']
    q_vec = tfidfvectorizer.transform(q)
    
    # Make prediction
    q_pred = LReg.predict(q_vec)

    return mlb.inverse_transform(q_pred)

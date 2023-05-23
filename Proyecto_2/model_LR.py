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
    le = MultiLabelBinarizer()
    tfidf_vectorizer = TfidfVectorizer()
    
    def clean_text(text):
        text = re.sub((r'[^\w\s]'),'', text).lower() 
        text = re.sub((r'\d+'),'', text).lower()
        text = re.sub((r'_+'),'', text).lower()
        wnl = nltk.stem.WordNetLemmatizer()
        stopwords = nltk.corpus.stopwords.words('english')
        words = re.sub(r'[^\w\s]', ' ', text).split()
        return ' '.join([wnl.lemmatize(word) for word in words if word not in stopwords])
    
    ####################################################################################################################
    
    # Carga de datos de archivo .csv
    dataTraining = pd.read_csv('https://github.com/albahnsen/MIAD_ML_and_NLP/raw/main/datasets/dataTraining.zip', encoding='UTF-8', index_col=0)
    dataTraining['clean_plot'] = dataTraining['plot'].apply(clean_text)
    
    xtrain_tfidf = tfidf_vectorizer.fit_transform(dataTraining['clean_plot'])
    xtrain_tfidf = tfidf_vectorizer.transform(dataTraining['clean_plot'])
    
    ###################################################################################################################
    
    df5000 = pd.DataFrame(columns=['plot'],index=range(1))
    df5000['plot'] = features
    df5000['clean_plot'] = df5000['plot'].apply(clean_text)
     
    q = df5000['clean_plot']
    q_vec = tfidf_vectorizer.transform(q)
    
    # Make prediction
    q_pred = regressor.predict(q_vec)

    return le.inverse_transform(q_pred)

import pandas as pd
import numpy as np
import joblib
import sys
import os

def predict_genre(features):
    
    regressor = joblib.load(os.path.dirname(__file__) + '/movie_genres_LR.pkl')
    
    dtypes = np.dtype([("plot", str)])
    df = pd.DataFrame(np.empty(0, dtype=dtypes))
    
    df['plot'] = str(features[0])
    
    # Make prediction
    p1 = regressor.predict(df)[0]

    return p1

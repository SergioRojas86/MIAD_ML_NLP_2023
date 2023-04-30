import pandas as pd
import joblib
import sys
import os

def predict_price(features):
    
    regressor = joblib.load(os.path.dirname(__file__) + '/car_listing_XGB.pkl')
    #regressor = joblib.load(os.path.dirname(__file__) + '\car_listing_XGB.pkl')
    
    dataTraining = []
    dataTraining = pd.read_csv('https://raw.githubusercontent.com/albahnsen/MIAD_ML_and_NLP/main/datasets/dataTrain_carListings.zip')
    dataTraining['State'] = dataTraining['State'].str.replace(" ", "")
    dataTraining['Model'] = dataTraining['Model'].str.replace(" ", "")
    dataTraining['Make'] = dataTraining['Make'].str.replace(" ", "")
    dTrain = pd.get_dummies(dataTraining)
    df1 = dTrain.iloc[[0],:]
    df1[df1 > 0]= 0
    
    State_ = 'State_'+str(features[2])
    Make_ = 'Make_'+str(features[3])
    Model_= 'Model_'+features[4]
    #
    df1['Year'] = int(features[0])
    df1['Mileage'] = int(features[1])
    df1[State_] = int(1)
    df1[Make_] = int(1)
    df1[Model_] = int(1)
    
    # Make prediction
    p1 = regressor.predict(df1.drop('Price', axis=1))[0]

    return p1

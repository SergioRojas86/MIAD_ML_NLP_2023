import pandas as pd
import joblib
import sys
import os

def predict_price(features):
    
    regressor = joblib.load(os.path.dirname(__file__) + '/car_listing_XGB.pkl')
    #regressor = joblib.load(os.path.dirname(__file__) + '\car_listing_XGB.pkl')
    print(regressor)
    
    dataTraining = pd.read_csv('https://raw.githubusercontent.com/albahnsen/MIAD_ML_and_NLP/main/datasets/dataTrain_carListings.zip')
    dataTraining['State'] = dataTraining['State'].str.replace(" ", "")
    dataTraining['Model'] = dataTraining['Model'].str.replace(" ", "")
    dataTraining['Make'] = dataTraining['Make'].str.replace(" ", "")
    dTrain =  dataTraining.copy()
    dTrain = pd.get_dummies(dTrain)
    columnas = dTrain.columns
    df1 = dTrain.iloc[[0],:]
    df1[df1 > 0]= 0
    #dTrain[dTrain > 0]= 0
    
    State_ = 'State_'+features[2]
    Make_ = 'Make_'+features[3]
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


if __name__ == "__main__":
    
    if len(sys.argv) == 1:
        print('Please add a Year, Mileage, State and brand of car')
        
    else:
        lst = sys.argv[1]

        p1 = predict_price([Year,Mileage,State,Make])
        
        print([Year,Mileage,State,Make])
        print('Probability of price: ', p1)
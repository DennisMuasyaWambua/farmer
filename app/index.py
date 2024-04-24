import pandas as pd
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import os



app = FastAPI()




# farm class
class Soil(BaseModel):
    N:float 
    P:float
    K:float
    temperature:float
    humidity:float
    ph:float
    rainfall:float



@app.post("/soil")
def preprocess_and_model(data:Soil):  
    # checkLambda()
    N = data.N
    P = data.P
    K = data.K
    temperature = data.temperature
    humidity = data.humidity
    ph = data.ph
    rainfall = data.rainfall
    data = [[N, P, K, temperature, humidity, ph, rainfall]]
    df = pd.DataFrame(data, columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
    full_pipeline = joblib.load('app/22-03-2023_17-12-14_full_pipeline.pkl')
    xgb_clf = joblib.load('app/22-03-2023_17-12-14_xgb_clf.pkl')


    prepared_data = full_pipeline.transform(df)
    prediction = xgb_clf.predict(prepared_data)

   
    target_encoder = joblib.load('app/22-03-2023_17-12-14_target_encoder.pkl')

    target_value = target_encoder.inverse_transform(prediction)

    print(target_value)
   

    return {"recommended": target_value[0]}


import pandas as pd
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import os
from mangum import Mangum


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
    full_pipeline = joblib.load('/home/dennis/Desktop/projects/farmers/app/saved_model/02-05-2023_20-47-39_full_pipeline.pkl')
    xgb_clf = joblib.load('/home/dennis/Desktop/projects/farmers/app/saved_model/02-05-2023_20-47-39_xgb_clf.pkl')


    prepared_data = full_pipeline.transform(df)
    prediction = xgb_clf.predict(prepared_data)

   
    target_encoder = joblib.load('/home/dennis/Desktop/projects/farmers/app/saved_model/02-05-2023_20-47-39_target_encoder.pkl')

    target_value = target_encoder.inverse_transform(prediction)

    print(target_value)
   

    return {"recommended": target_value[0]}

handler = Mangum(app)

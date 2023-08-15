import pandas as pd
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from supabase_py import create_client



load_dotenv()
app = FastAPI()

url = os.getenv("SUPABASE_URL")
key = os.getenv("ANON")

client = create_client(url, key)

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
    N = data.N
    P = data.P
    K = data.K
    temperature = data.temperature
    humidity = data.humidity
    ph = data.ph
    rainfall = data.rainfall
    data = [[N, P, K, temperature, humidity, ph, rainfall]]
    df = pd.DataFrame(data, columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
    full_pipeline = joblib.load('saved_model/02-05-2023_20-47-39_full_pipeline.pkl')
    xgb_clf = joblib.load('saved_model/02-05-2023_20-47-39_xgb_clf.pkl')


    prepared_data = full_pipeline.transform(df)
    prediction = xgb_clf.predict(prepared_data)

   
    target_encoder = joblib.load('saved_model/02-05-2023_20-47-39_target_encoder.pkl')

    target_value = target_encoder.inverse_transform(prediction)

    print(target_value)
    print(client.rest_url)

    return {"recommended": target_value[0]}



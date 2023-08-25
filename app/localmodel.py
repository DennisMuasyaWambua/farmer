import pandas as pd
import joblib
import numpy as np

data = [[10,40,50,30,80,3,100000]]
df = pd.DataFrame(data, columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])


def soil():
    full_pipeline = joblib.load('/home/dennis/Desktop/projects/farmers/app/saved_model/02-05-2023_20-47-39_full_pipeline.pkl')
    xgb_clf = joblib.load('/home/dennis/Desktop/projects/farmers/app/saved_model/02-05-2023_20-47-39_xgb_clf.pkl')

    prepared_data = full_pipeline.transform(df)
    prediction = xgb_clf.predict(prepared_data)

    target_encoder = joblib.load('/home/dennis/Desktop/projects/farmers/app/saved_model/02-05-2023_20-47-39_target_encoder.pkl')

    target_value = target_encoder.inverse_transform(prediction)

    print(target_value[0])
    print(type(target_value))
    target_string = np.array_str(target_value)
    print(target_string)
    return target_value

soil()
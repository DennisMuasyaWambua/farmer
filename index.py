import pandas as pd
import joblib

data = [[90, 42, 43, 20.879744,	82.002744, 6.502985, 202]]

# Create the pandas DataFrame
df = pd.DataFrame(data, columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])

print(df)


def preprocess_and_model(df):
    full_pipeline = joblib.load('02-05-2023_20-47-39_full_pipeline.pkl')
    xgb_clf = joblib.load('02-05-2023_20-47-39_xgb_clf.pkl')

    prepared_data = full_pipeline.transform(df)
    prediction = xgb_clf.predict(prepared_data)

    target_encoder = joblib.load('02-05-2023_20-47-39_target_encoder.pkl')

    target_value = target_encoder.inverse_transform(prediction)

    print(target_value)

    return target_value

preprocess_and_model(df)


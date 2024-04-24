import pandas as pd
import joblib

data = [[105, 10, 20, 	20.417112,	100.636362, 	8.086922, 	100]]

# Create the pandas DataFrame
df = pd.DataFrame(data, columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])

print(df)


def preprocess_and_model(df):
    full_pipeline = joblib.load('/home/dennis/Desktop/farmers/22-03-2023_17-12-14_full_pipeline.pkl')
    xgb_clf = joblib.load('/home/dennis/Desktop/farmers/22-03-2023_17-12-14_xgb_clf.pkl')

    prepared_data = full_pipeline.transform(df)
    prediction = xgb_clf.predict(prepared_data)

    target_encoder = joblib.load('/home/dennis/Desktop/farmers/22-03-2023_17-12-14_target_encoder.pkl')

    target_value = target_encoder.inverse_transform(prediction)

    # print(target_value)

    return target_value

preprocess_and_model(df)

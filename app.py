from fastapi import FastAPI
import pickle
import numpy as np

app = FastAPI()

# load train model
model = pickle.load(open("model/model.pkl", 'rb'))

@app.get("/")
def home():
    return {"message": "Welcome to the delivery time prediction API!!! API running successfully."}

@app.post("/predict")
def predict(data: dict):
    features = np.array([[ 
    data["Delivery_person_Age"],
    data["Delivery_person_Ratings"],
    data["Restaurant_latitude"],
    data["Restaurant_longitude"],
    data["Delivery_location_latitude"],
    data["Delivery_location_longitude"],
    data["Vehicle_condition"],
    data["multiple_deliveries"],
    data["order_hour"],
    data["pickup_hour"],
    data["prep_time"] 
    ]])
    prediction = model.predict(features)
    return {"predicted_time": float(prediction[0])}
import cv2
import numpy as np
import joblib
import pandas as pd
from tensorflow.keras.models import load_model

# ===== Load Model & Scaler & Encoder =====
model = load_model("building_predictor_cnn.h5")
scaler = joblib.load("floor_predictor_scaler.pkl")
le = joblib.load("encoder.pkl")

IMG_SIZE = 128

# ===== Prediction Function =====
def predict_building(image_path, building_type):
    # Image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Tabular
    building_encoded = le.transform([building_type])
    tab_scaled = scaler.transform(building_encoded.reshape(1, -1))

    # Predict
    pred = model.predict([img, tab_scaled])
    width_m, height_m, floors = pred[0]
    floors = int(round(floors))

    return {
        "image_path": image_path,
        "width_m": round(width_m, 2),
        "height_m": round(height_m, 2),
        "estimated_floors": floors
    }

# ===== Example Usage =====
file_path = "images/29.jpg"
building_type = "Residential (Low-rise)"

result = predict_building(file_path, building_type)
print("Prediction Result:")
print(result)

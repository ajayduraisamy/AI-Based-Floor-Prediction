# ===== 0. Import Libraries =====
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras import layers, models, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib
import tensorflow as tf

# ===== 1. Load Dataset =====
df = pd.read_csv("dataset.csv")

# Encode building type
le = LabelEncoder()
df["building_type_encoded"] = le.fit_transform(df["building_type"])
joblib.dump(le, "encoder.pkl")
print("Encoder saved as encoder.pkl")

# Tabular features
X_tab = df[["building_type_encoded"]].values  

# Targets: width_m, height_m, estimated_floors
y = df[["width_m", "height_m", "estimated_floors"]].values  

# ===== 2. Scale Tabular Features =====
scaler = StandardScaler()
X_tab_scaled = scaler.fit_transform(X_tab)
joblib.dump(scaler, "floor_predictor_scaler.pkl")
print("Scaler saved as floor_predictor_scaler.pkl")

# Optional: Scale targets
y_scaler = StandardScaler()
y_scaled = y_scaler.fit_transform(y)
joblib.dump(y_scaler, "y_scaler.pkl")
print("Target scaler saved as y_scaler.pkl")

# ===== 3. Image Preprocessing =====
IMG_SIZE = 128
image_data = []

for path in df["image_path"]:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    image_data.append(img)

X_img = np.array(image_data)

# ===== 4. Train-Test Split =====
X_img_train, X_img_test, X_tab_train, X_tab_test, y_train, y_test = train_test_split(
    X_img, X_tab_scaled, y_scaled, test_size=0.2, random_state=42
)

# ===== 5. Data Augmentation =====
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(X_img_train)

# ===== 6. Build Smaller CNN for Images =====
cnn_input = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = layers.Conv2D(16, (3,3), activation="relu")(cnn_input)
x = layers.MaxPooling2D(2,2)(x)
x = layers.Conv2D(32, (3,3), activation="relu")(x)
x = layers.MaxPooling2D(2,2)(x)
x = layers.Conv2D(64, (3,3), activation="relu")(x)
x = layers.MaxPooling2D(2,2)(x)
x = layers.Flatten()(x)
x = layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
x = layers.Dropout(0.5)(x)

# ===== 7. Tabular Input =====
tab_input = Input(shape=(X_tab_train.shape[1],))
y_tab = layers.Dense(16, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001))(tab_input)

# ===== 8. Merge CNN + Tabular =====
merged = layers.concatenate([x, y_tab])
z = layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001))(merged)
z = layers.Dropout(0.3)(z)
z = layers.Dense(32, activation="relu")(z)

# ===== 9. Output (Multi-Regression) =====
output = layers.Dense(3)(z)  # Predict width_m, height_m, estimated_floors

model = models.Model(inputs=[cnn_input, tab_input], outputs=output)
model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.summary()

# ===== 10. Callbacks =====
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

# ===== 11. Train Model =====
history = model.fit(
    datagen.flow([X_img_train, X_tab_train], y_train, batch_size=8),
    validation_data=([X_img_test, X_tab_test], y_test),
    epochs=200,
    callbacks=[early_stop, reduce_lr]
)

# ===== 12. Evaluate =====
y_pred_scaled = model.predict([X_img_test, X_tab_test])
y_pred = y_scaler.inverse_transform(y_pred_scaled)
y_test_orig = y_scaler.inverse_transform(y_test)

rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred))
r2 = r2_score(y_test_orig, y_pred)
print("RMSE:", rmse)
print("R²:", r2)

# ===== 13. Save Model =====
model.save("building_predictor_cnn.keras")
print("Model saved as 'building_predictor_cnn.keras'")

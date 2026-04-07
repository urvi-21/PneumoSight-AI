import tensorflow as tf
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import gdown

MODEL_PATH = "models/pneumonia_model.keras"
def download_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs("models", exist_ok=True)
        url = "https://drive.google.com/file/d/1RAe7YQ9chy-nq8wvHoxk2aMM2Hhy62tT"
        gdown.download(url, MODEL_PATH, quiet=False)

model = None

def load_model():
    global model
    if model is None:
        download_model()
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model

def preprocess(img):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    img = img.resize((224, 224))
    img = np.array(img).astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

def predict(img):
    model = load_model()
    img_array = preprocess(img)

    prob = model.predict(img_array)[0][0]
    label = "Pneumonia" if prob > 0.5 else "Normal"

    return label, float(prob)
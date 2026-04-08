import tensorflow as tf
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import gdown

MODEL_PATH = "models/pneumonia_model.keras"

def download_model():
    url = "https://drive.google.com/uc?id=1RAe7YQ9chy-nq8wvHoxk2aMM2Hhy62tT"

    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 10000000:
        print("Downloading model...")
        gdown.download(url, MODEL_PATH, quiet=False)

def load_model():
    download_model()
    return tf.keras.models.load_model(MODEL_PATH, compile=False)
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
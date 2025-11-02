# model.py  -- lightweight runtime helpers for inference
import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Paths (adjust if you keep model in a different place)
MODEL_PATH = os.path.join("model", "emotion_model.h5")
CLASS_FILE = os.path.join("model", "class_names.txt")

# Load class names (one per line). If missing, fallback to common FER order.
def _load_class_names(path=CLASS_FILE):
    if os.path.exists(path):
        with open(path, "r") as f:
            classes = [l.strip() for l in f.readlines() if l.strip()]
            if classes:
                return classes
    # fallback common FER ordering
    return ["Anger", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"]

CLASS_NAMES = _load_class_names()

# Load model once at import time
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Place your emotion_model.h5 there.")
MODEL = load_model(MODEL_PATH)

# Preprocess bytes -> model input
def _preprocess_image_bytes(image_bytes, target_size=(48,48), to_gray=True):
    nparr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode image")
    if to_gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, target_size)
    arr = img.astype("float32") / 255.0
    if to_gray:
        arr = np.expand_dims(arr, axis=-1)   # (h,w,1)
    arr = np.expand_dims(arr, axis=0)       # (1,h,w,c)
    return arr

# Public helper: give image bytes, return (label, confidence, raw_probs)
def predict_image(image_bytes):
    """
    image_bytes: raw bytes from uploaded file (request.files['file'].read())
    returns: (label:str, confidence:float, probs: np.ndarray)
    """
    x = _preprocess_image_bytes(image_bytes)
    probs = MODEL.predict(x)[0]
    idx = int(np.argmax(probs))
    label = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else str(idx)
    confidence = float(probs[idx])
    return label, confidence, probs

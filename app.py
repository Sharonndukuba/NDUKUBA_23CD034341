# app.py
from flask import Flask, request, render_template, jsonify
import numpy as np
import cv2, base64, os, sqlite3, datetime
from tensorflow.keras.models import load_model

app = Flask(__name__)
MODEL_PATH = "model/emotion_model.h5"
EMOTIONS = ['angry','disgust','fear','happy','sad','surprise','neutral']

# Load model at startup
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"{MODEL_PATH} not found. Train model or place model/emotion_model.h5 before running.")
model = load_model(MODEL_PATH)

# face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

DB_PATH = "database.db"
def ensure_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS records
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, timestamp TEXT, image_path TEXT, emotion TEXT, confidence REAL)''')
    conn.commit(); conn.close()

def save_record(name, image_bytes, emotion, confidence):
    ensure_db()
    ts = datetime.datetime.utcnow().isoformat()
    os.makedirs('captures', exist_ok=True)
    # sanitize filename
    fname = f"captures/{ts.replace(':','-').replace('.','-')}.png"
    with open(fname, 'wb') as f:
        f.write(image_bytes)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO records (name,timestamp,image_path,emotion,confidence) VALUES (?,?,?,?,?)",
              (name, ts, fname, emotion, float(confidence)))
    conn.commit(); conn.close()

def decode_base64_image(data_url):
    header, encoded = data_url.split(',', 1)
    data = base64.b64decode(encoded)
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img, data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        content = request.get_json()
        img_b64 = content.get('image')
        name = content.get('name', 'anonymous')
        if not img_b64:
            return jsonify({'error': 'No image provided'}), 400
        img, raw_bytes = decode_base64_image(img_b64)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(faces) == 0:
            face = cv2.resize(gray, (48,48))
        else:
            (x,y,w,h) = faces[0]
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48,48))
        face_rgb = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB).astype('float32') / 255.0
        face_rgb = np.expand_dims(face_rgb, axis=0)
        preds = model.predict(face_rgb)[0]
        idx = int(np.argmax(preds))
        emotion = EMOTIONS[idx]
        confidence = float(preds[idx])
        save_record(name, raw_bytes, emotion, confidence)
        return jsonify({'emotion': emotion, 'confidence': confidence})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    ensure_db()
    # For production use gunicorn; locally you can use debug True
    app.run(host='0.0.0.0', port=5000, debug=True)

from fastapi import FastAPI, File, UploadFile
import librosa
import numpy as np
import joblib
from collections import Counter
import io

app = FastAPI()

# Paths
model_path = "xgboost_heart_sound_model.pkl"
scaler_path = "scaler_heart_sound.pkl"

# Load model and scaler
clf_xgb = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Since model was trained on 5s chunks, split the audio into 5s overlapping windows
def sliding_window_chunks(y, sr, window_sec=5, stride_sec=1):
    window_size = int(window_sec * sr)
    stride_size = int(stride_sec * sr)
    chunks = []
    for start in range(0, len(y) - window_size + 1, stride_size):
        chunk = y[start:start+window_size]
        chunks.append(chunk)
    return chunks

# Extract MFCC features for each chunk
def extract_features(audio_chunks, sr, n_mfcc=13):
    features = []
    for chunk in audio_chunks:
        mfcc = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        feature_vector = np.concatenate([mfcc_mean, mfcc_std])
        features.append(feature_vector)
    return np.array(features)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Load audio
    audio_bytes = await file.read()
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)

    chunks = sliding_window_chunks(y, sr, window_sec=5, stride_sec=1)

    X_new = extract_features(chunks, sr)

    # Scale features
    X_new_scaled = scaler.transform(X_new)

    # Predict
    y_pred_chunks = clf_xgb.predict(X_new_scaled)

    # Majority vote over all chunks
    majority_label = Counter(y_pred_chunks).most_common(1)[0][0]

    label_map = {0: "Healthy", 1: "Unhealthy"}
    prediction = label_map[majority_label]

    return {"prediction": prediction}

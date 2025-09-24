from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import librosa
import numpy as np
import joblib
from collections import Counter
import io

# Metadata for the API
description = """
This API classifies heart sounds as **Healthy** or **Unhealthy** using an XGBoost model.

## How to Use

1.  Upload a `.wav` audio file of a heart sound.
2.  The API will process the audio, make a prediction, and return the classification.
"""

app = FastAPI(
    title="Heart Sound Classification API",
    description=description,
    version="1.0.0",
)

# Define the response model
class PredictionResponse(BaseModel):
    prediction: str

# Paths
model_path = "xgboost_heart_sound_model.pkl"
scaler_path = "scaler_heart_sound.pkl"

# Load model and scaler
try:
    clf_xgb = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
except FileNotFoundError:
    raise RuntimeError("Model or scaler not found. Make sure to train the model first.")

# Since model was trained on 5s chunks, split the audio into 5s overlapping windows
def sliding_window_chunks(y, sr, window_sec=5, stride_sec=1):
    window_size = int(window_sec * sr)
    stride_size = int(stride_sec * sr)
    chunks = []
    if len(y) < window_size:
        # Pad the audio if it's shorter than the window size
        padding = window_size - len(y)
        y = np.pad(y, (0, padding), 'constant')
    
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

@app.post("/predict/", 
          response_model=PredictionResponse,
          summary="Classify a heart sound",
          description="Upload a `.wav` audio file to classify it as Healthy or Unhealthy.")
async def predict(file: UploadFile = File(..., description="Heart sound audio file (.wav)")):
    # Verify file type
    if not file.filename.endswith(".wav"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a .wav file.")

    # Load audio
    try:
        audio_bytes = await file.read()
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not process audio file: {e}")

    chunks = sliding_window_chunks(y, sr, window_sec=5, stride_sec=1)
    if not chunks:
        raise HTTPException(status_code=400, detail="Audio file is too short to be processed.")

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

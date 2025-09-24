import kagglehub
import librosa
import numpy as np
import os
from tqdm import tqdm 
import librosa
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
from xgboost import XGBClassifier
import joblib
from sklearn.preprocessing import StandardScaler
database_path = kagglehub.dataset_download('swapnilpanda/heart-sound-database')
print('Data source import complete.')

audio_files = {
    "healthy_train": [],
    "unhealthy_test": []
}

base_path = database_path

# Collect healthy/train files
healthy_train_path = os.path.join(base_path, "heart_sound", "train", "healthy")
if os.path.exists(healthy_train_path):
    for filename in os.listdir(healthy_train_path):
        if filename.endswith(".wav"):
            audio_files["healthy_train"].append(os.path.join(healthy_train_path, filename))

# Collect unhealthy/train files
unhealthy_train_path = os.path.join(base_path, "heart_sound", "train", "unhealthy")
if os.path.exists(unhealthy_train_path):
    for filename in os.listdir(unhealthy_train_path):
        if filename.endswith(".wav"):
            audio_files["unhealthy_test"].append(os.path.join(unhealthy_train_path, filename))

print(f"Healthy train files found: {len(audio_files['healthy_train'])}")
print(f"Unhealthy test files found: {len(audio_files['unhealthy_test'])}")

# Load and inspect one healthy/train file
if audio_files["healthy_train"]:
    first_healthy = audio_files["healthy_train"][0]
    print(f"\nLoading healthy/train file")
    y, sr = librosa.load(first_healthy, sr=None)
    print(f"Audio shape: {y.shape}, Sampling rate: {sr} Hz")
        

if audio_files["unhealthy_test"]:
    first_unhealthy = audio_files["unhealthy_test"][0]
    print(f"\nLoading unhealthy/test file")
    y, sr = librosa.load(first_unhealthy, sr=None)
    print(f"Audio shape: {y.shape}, Sampling rate: {sr} Hz")
else:
    print("No unhealthy/test files found.")


def get_durations(file_list):
    """
    Calculate the duration of audio files in a list.

    Args:
        file_list (list): A list of file paths to audio files.

    Returns:
        np.ndarray: An array of durations in seconds for each audio file.
    """
    durations = []
    for file_path in file_list:
        try:
            y, sr = librosa.load(file_path, sr=None)
            duration = len(y) / sr
            durations.append(duration)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    return np.array(durations)

def summarize_durations(durations, label):
    """
    Summarize audio file durations and print statistics.

    Args:
        durations (np.ndarray): An array of audio file durations.
        label (str): A label for the dataset (e.g., "Healthy", "Unhealthy").
    """
    print(f"\n--- {label} ---")
    if len(durations) == 0:
        print("No files found.")
        return
    unique_durations = np.unique(np.round(durations, 2))  # round for near-identical cases
    if len(unique_durations) == 1:
        print(f"All samples have the same duration: {unique_durations[0]} seconds")
    else:
        print(f"Number of files: {len(durations)}")
        print(f"Min duration: {durations.min():.2f} sec")
        print(f"Max duration: {durations.max():.2f} sec")
        print(f"Mean duration: {durations.mean():.2f} sec")
        print(f"Median duration: {np.median(durations):.2f} sec")
        print(f"Unique durations: {len(unique_durations)}")
        print(f"Some unique values: {unique_durations[:10]}{'...' if len(unique_durations)>10 else ''}")

def compute_snr(y):
    """
    Compute approximate Signal-to-Noise Ratio (SNR) of an audio signal in dB.

    Args:
        y (np.ndarray): The audio signal.

    Returns:
        float: The SNR in dB. Returns infinity if noise power is zero.
    """
    signal_power = np.mean(y ** 2)
    noise_power = np.var(y)
    if noise_power == 0:
        return float("inf")
    return 10 * np.log10(signal_power / noise_power)

def analyze_snr(file_list, label):
    """
    Analyze the Signal-to-Noise Ratio (SNR) for a list of audio files.

    Args:
        file_list (list): A list of file paths to audio files.
        label (str): A label for the dataset (e.g., "Healthy", "Unhealthy").

    Returns:
        tuple: A tuple containing:
            - list: Filtered list of (file_path, snr) tuples for files within 2 std devs of the mean SNR.
            - np.ndarray: Array of all computed SNR values.
    """
    snr_values = []

    for file_path in tqdm(file_list, desc=f"Computing SNR for {label}"):
        try:
            y, sr = librosa.load(file_path, sr=None)
            snr = compute_snr(y)
            snr_values.append((file_path, snr))
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    snr_array = np.array([val for _, val in snr_values])
    mean_snr = np.mean(snr_array)
    std_snr = np.std(snr_array)
    min_snr = np.min(snr_array)
    max_snr = np.max(snr_array)

    print(f"\n--- {label} ---")
    print(f"Files: {len(snr_array)}")
    print(f"Min SNR: {min_snr:.2f} dB")
    print(f"Max SNR: {max_snr:.2f} dB")
    print(f"Mean SNR: {mean_snr:.2f} dB")
    print(f"Std Dev: {std_snr:.2f} dB")

    lower_bound = mean_snr - 2 * std_snr
    upper_bound = mean_snr + 2 * std_snr
    filtered = [(fp, snr) for fp, snr in snr_values if lower_bound <= snr <= upper_bound]

    print(f"Outliers removed: {len(snr_array) - len(filtered)}")
    print(f"Remaining files: {len(filtered)}")

    return filtered, snr_array

healthy_filtered, healthy_snr = analyze_snr(audio_files["healthy_train"], "Healthy (train)")
unhealthy_filtered, unhealthy_snr = analyze_snr(audio_files["unhealthy_test"], "Unhealthy (test)")

healthy_filtered_files = [fp for fp, _ in healthy_filtered]
unhealthy_filtered_files = [fp for fp, _ in unhealthy_filtered]


def sliding_window_chunks(y, sr, window_sec=5, stride_sec=1):
    """
    Split an audio signal into overlapping sub-samples.

    Args:
        y (np.ndarray): Audio signal.
        sr (int): Sampling rate.
        window_sec (int): Duration of each chunk in seconds.
        stride_sec (int): Step size between chunks in seconds.

    Returns:
        List of np.ndarray: List of audio chunks.
    """
    window_size = int(window_sec * sr)
    stride_size = int(stride_sec * sr)
    chunks = []

    for start in range(0, len(y) - window_size + 1, stride_size):
        end = start + window_size
        chunk = y[start:end]
        chunks.append(chunk)

    return chunks

healthy_processed_audio = []
unhealthy_processed_audio = []

print("Processing Healthy (train) samples...")
for file_path in tqdm(healthy_filtered_files):
    try:
        y, sr = librosa.load(file_path, sr=None)
        chunks = sliding_window_chunks(y, sr, window_sec=5, stride_sec=1)
        healthy_processed_audio.extend(chunks)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")

print("Processing Unhealthy (test) samples...")
for file_path in tqdm(unhealthy_filtered_files):
    try:
        y, sr = librosa.load(file_path, sr=None)
        chunks = sliding_window_chunks(y, sr, window_sec=5, stride_sec=1)
        unhealthy_processed_audio.extend(chunks)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")

print(f"\nTotal processed healthy chunks: {len(healthy_processed_audio)}")
print(f"Total processed unhealthy chunks: {len(unhealthy_processed_audio)}")


def extract_features(audio_chunks, sr=22050, n_mfcc=13):
    """
    Extract Mel-Frequency Cepstral Coefficients (MFCCs) from audio chunks.

    Args:
        audio_chunks (list): A list of audio chunks (np.ndarray).
        sr (int): The sampling rate of the audio. Defaults to 22050.
        n_mfcc (int): The number of MFCCs to extract. Defaults to 13.

    Returns:
        np.ndarray: An array of extracted features, where each row represents a chunk.
    """
    features = []
    for y in tqdm(audio_chunks, desc="Extracting MFCCs"):
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        feature_vector = np.concatenate([mfcc_mean, mfcc_std])
        features.append(feature_vector)
    return np.array(features)

# 1. Extract features
print("Extracting features for Healthy...")
X_healthy = extract_features(healthy_processed_audio)

print("Extracting features for Unhealthy...")
X_unhealthy = extract_features(unhealthy_processed_audio)

X = np.vstack([X_healthy, X_unhealthy])
y = np.array([0]*len(X_healthy) + [1]*len(X_unhealthy))  # 0=healthy, 1=unhealthy

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("\n=== XGBoost ===")
clf_xgb = XGBClassifier(
    n_estimators=400,         # number of boosting rounds
    max_depth=8,              # tree depth
    learning_rate=0.05,       # smaller LR with more estimators
    subsample=0.8,            # random subsampling for robustness
    colsample_bytree=0.8,     # feature subsampling
    scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]), # handle imbalance
    eval_metric="logloss",
    n_jobs=-1,
    random_state=42
)
clf_xgb.fit(X_train, y_train)
y_pred_xgb = clf_xgb.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb))

# Path to save the model
model_path = "xgboost_heart_sound_model.pkl"
scaler_path = "scaler_heart_sound.pkl"
# Save the trained XGBoost model
joblib.dump(clf_xgb, model_path)
print(f"Model saved to {model_path}")
joblib.dump(scaler, scaler_path)
print(f"Scaler saved to {scaler_path}")

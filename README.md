# Heart Sound Classification Project

This project is designed to classify heart sounds as either "healthy" or "unhealthy" using a machine learning model. It encompasses a full machine learning pipeline, from data exploration and preprocessing to model training and API deployment.

## File Structure

```
Heart-Condition-Detector/
├── Analysis_And_Training/
│   ├── data_exploration.ipynb
│   ├── ml_model_building.ipynb
│   ├── model_train.py
│   ├── requirement.txt
│   └── ... (other files)
├── API/
│   ├── Dockerfile
│   ├── main.py
│   ├── requirements.txt
│   └── ... (other files)
└── README.md
```

## Analysis and Training

The `Analysis_And_Training/` directory contains all the necessary scripts and notebooks for data exploration, preprocessing, and model training.

*   **`data_exploration.ipynb`**: This notebook handles the initial data loading, inspection, and preprocessing of the heart sound audio files. It visualizes the audio data using waveforms and spectrograms to understand its properties. The notebook then segments the audio into uniform 5-second chunks and saves them for model building.

*   **`ml_model_building.ipynb`**: This notebook focuses on feature extraction, model training, and evaluation. It extracts Mel-Frequency Cepstral Coefficients (MFCCs) from the preprocessed audio chunks. It also addresses class imbalance and compares several classification models, ultimately selecting XGBoost as the best performer.

*   **`model_train.py`**: This script provides a streamlined, end-to-end pipeline for training the model. It automates the data loading, preprocessing, feature extraction, and model training steps. The final trained model and scaler are saved to disk for later use in the API.

## API

The `API/` directory contains the code for deploying the trained model as a RESTful API using FastAPI. This allows for real-time predictions on new heart sound audio files.

## How to Run

### Training the Model

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/abdullah7795/Heart-Condition-Detector.git
    cd Heart-Condition-Detector
    ```

2.  **Create a virtual environment and install dependencies**:
    ```bash
    python3 -m venv Analysis_And_Training/.menv
    source Analysis_And_Training/.menv/bin/activate
    pip install -r Analysis_And_Training/requirement.txt
    ```

3.  **Run the training script**:
    ```bash
    python3 Analysis_And_Training/model_train.py
    ```

### Running the API

1.  **Create a virtual environment and install dependencies**:
    ```bash
    python3 -m venv API/.venv
    source API/.venv/bin/activate
    pip install -r API/requirements.txt
    ```

2.  **Run the API server**:
    ```bash
    uvicorn API.main:app --reload --port 8001
    ```
    You can access the API documentation at `http://127.0.0.1:8001/docs`.

### Running the API with Docker

1.  **Navigate to the `API` directory**:
    ```bash
    cd API
    ```

2.  **Build the Docker image**:
    ```bash
    sudo docker build -t heart-sound-api .
    ```

3.  **Run the Docker container**:
    ```bash
    sudo docker run -p 8031:80 heart-sound-api
    ```
    The API will be accessible at `http://127.0.0.1:8031/docs`.
>>>>>>> main

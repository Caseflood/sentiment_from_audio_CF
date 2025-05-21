# Audio Emotion Recognition System

A comprehensive system for real-time emotion recognition from audio using OpenSMILE features and deep neural networks.

## Features

- Real-time emotion prediction from microphone input
- Support for 7 emotions: angry, disgust, fear, happy, neutral, sad, surprised
- Uses OpenSMILE for robust audio feature extraction
- Deep neural network model with high accuracy
- Twilio integration for phone call emotion analysis
- Environment variable configuration for secure deployment

## System Requirements

- Python 3.8+
- OpenSMILE
- ffmpeg
- CUDA-compatible GPU (optional, for faster training)
- Twilio account (for phone call analysis)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
pip install -r requirements_twilio.txt  # For Twilio integration
```

3. Install system dependencies:
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y ffmpeg opensmile

# macOS
brew install ffmpeg opensmile
```

4. Set up environment variables:
Create a `.env` file in the project root with the following variables:
```bash
# Twilio credentials
TWILIO_ACCOUNT_SID=your_account_sid_here
TWILIO_AUTH_TOKEN=your_auth_token_here
TWILIO_PHONE_NUMBER=your_twilio_phone_number_here

# Server configuration
SERVER_URL=https://your-ngrok-url.ngrok-free.app
PORT=5000

# Audio parameters
SAMPLE_RATE=16000
CHANNELS=1

# Model parameters
MODEL_PATH=emotion_model.keras
DATASET_PATH=emotion_dataset.csv

# OpenSMILE configuration
FEATURE_SET=ComParE_2016
FEATURE_LEVEL=Functionals
```

## Components

### 1. Feature Extraction (`extract_features.py`)
- Uses OpenSMILE to extract audio features
- Supports ComParE_2016 feature set
- Processes audio files in WAV format

### 2. Model Training (`train_model.py`)
- Trains a deep neural network on extracted features
- Uses TensorFlow/Keras
- Includes data augmentation and validation
- Supports GPU acceleration
- Integrates with Weights & Biases for experiment tracking

### 3. Real-time Prediction (`predict_mic.py`)
- Captures audio from microphone
- Processes audio in real-time
- Displays top 3 emotions with confidence scores
- Uses a 5-second sliding window for analysis

### 4. File-based Prediction (`predict_emotion.py`)
- Analyzes emotions from audio files
- Supports batch processing
- Outputs detailed confidence scores

### 5. Twilio Integration (`twilio_emotion.py`)
- Handles incoming phone calls
- Records audio from callers
- Analyzes emotions in real-time
- Provides voice feedback of results
- Uses environment variables for secure configuration

## Usage

### Training the Model
```bash
python train_model.py
```

### Real-time Microphone Analysis
```bash
python predict_mic.py
```

### Analyzing Audio Files
```bash
python predict_emotion.py path/to/audio.wav
```

### Setting up Twilio Integration
1. Start the Flask server:
```bash
python twilio_emotion.py
```

2. Start ngrok to expose your local server:
```bash
ngrok http 5000
```

3. Update your `.env` file with the new ngrok URL

4. Configure your Twilio phone number's voice webhook to point to your ngrok URL + `/voice`

5. Make a call to your Twilio number to test the system

## Model Performance

The model achieves the following performance metrics on the test set:
- Overall accuracy: ~70%
- Best performance: Surprised (84% F1-score)
- Worst performance: Sad (54% F1-score)

## Directory Structure

```
.
├── README.md
├── requirements.txt
├── requirements_twilio.txt
├── .env
├── extract_features.py
├── train_model.py
├── predict_mic.py
├── predict_emotion.py
├── twilio_emotion.py
├── emotion_model.keras
├── emotion_dataset.csv
└── recordings/
    └── *.wav
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

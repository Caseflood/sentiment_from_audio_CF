import os
import numpy as np
import tensorflow as tf
import pandas as pd
from opensmile import Smile, FeatureSet, FeatureLevel
from sklearn.preprocessing import StandardScaler
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse, Gather
from flask import Flask, request, Response
import tempfile
import soundfile as sf
import io
import base64
import json
import requests
import urllib.request
from pydub import AudioSegment
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Force TensorFlow to use CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Twilio credentials from environment variables
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER')

# Server configuration from environment variables
SERVER_URL = os.getenv('SERVER_URL')
PORT = int(os.getenv('PORT', 5000))

# Audio parameters from environment variables
SAMPLE_RATE = int(os.getenv('SAMPLE_RATE', 16000))
CHANNELS = int(os.getenv('CHANNELS', 1))

# Model parameters from environment variables
MODEL_PATH = os.getenv('MODEL_PATH', 'emotion_model.keras')
DATASET_PATH = os.getenv('DATASET_PATH', 'emotion_dataset.csv')

# OpenSMILE configuration from environment variables
FEATURE_SET = os.getenv('FEATURE_SET', 'ComParE_2016')
FEATURE_LEVEL = os.getenv('FEATURE_LEVEL', 'Functionals')

app = Flask(__name__)
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

def download_audio(url, auth):
    """Download audio file from Twilio"""
    print(f"Downloading audio from: {url}")
    response = requests.get(url, auth=auth)
    if response.status_code == 200:
        return response.content
    else:
        raise Exception(f"Failed to download audio: {response.status_code}")

def convert_mp3_to_wav(mp3_data):
    """Convert MP3 data to WAV format"""
    print("Converting MP3 to WAV")
    # Save MP3 to temporary file
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as mp3_file:
        mp3_file.write(mp3_data)
        mp3_file.flush()
        
        # Convert to WAV
        audio = AudioSegment.from_mp3(mp3_file.name)
        wav_data = audio.export(format='wav')
        wav_bytes = wav_data.read()
        
        # Clean up
        os.unlink(mp3_file.name)
        wav_data.close()
        
        print("Conversion successful")
        return wav_bytes

def extract_features(audio_data):
    """Extract features from audio data using OpenSMILE"""
    print("Starting feature extraction")
    # Initialize OpenSMILE
    smile = Smile(
        feature_set=getattr(FeatureSet, FEATURE_SET),
        feature_level=getattr(FeatureLevel, FEATURE_LEVEL),
    )
    
    # Save audio to temporary file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        temp_file.write(audio_data)
        temp_file.flush()
        try:
            # Extract features
            features = smile.process_file(temp_file.name)
            features_array = features.values.flatten()
            print("Feature extraction successful")
            return features_array.reshape(1, -1)
        finally:
            os.unlink(temp_file.name)

def load_model_and_scaler():
    """Load the trained model and scaler"""
    print("Loading model and scaler...")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    df = pd.read_csv(DATASET_PATH)
    X = df.drop('emotion', axis=1).values
    scaler = StandardScaler()
    scaler.fit(X)
    
    emotions = sorted(df['emotion'].unique())
    print("Model and scaler loaded successfully")
    return model, scaler, emotions

# Load model and scaler at startup
model, scaler, emotions = load_model_and_scaler()

def predict_emotion(audio_url):
    """Predict emotion from audio URL"""
    try:
        print(f"Processing audio from URL: {audio_url}")
        # Download the audio file
        mp3_data = download_audio(audio_url, (TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN))
        print("Audio downloaded successfully")
        
        # Convert MP3 to WAV
        wav_data = convert_mp3_to_wav(mp3_data)
        
        # Extract features
        features = extract_features(wav_data)
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled, verbose=0)
        
        # Get top 3 emotions
        top_3_indices = np.argsort(prediction[0])[-3:][::-1]
        results = []
        for idx in top_3_indices:
            confidence = float(prediction[0][idx])
            if confidence > 0.05:  # Only include emotions with >5% confidence
                results.append({
                    'emotion': emotions[idx],
                    'confidence': confidence
                })
        
        print(f"Prediction results: {results}")
        return results
    except Exception as e:
        print(f"Error predicting emotion: {str(e)}")
        return None

@app.route("/voice", methods=['GET', 'POST'])
def voice():
    """Handle incoming voice calls"""
    print("\n=== Incoming Call ===")
    print(f"Method: {request.method}")
    print(f"Headers: {dict(request.headers)}")
    print(f"Form data: {request.form}")
    print(f"Query params: {request.args}")
    print("===================\n")
    
    response = VoiceResponse()
    
    # First, ask the user to speak
    response.say("Please speak for emotion analysis. After you finish speaking, stay on the line for the results.", voice='Polly.Amy')
    
    # Record the audio
    response.record(
        action='/process_audio',
        method='POST',
        maxLength='10',
        playBeep=False,
        trim='trim-silence',
        recordingStatusCallback='/process_audio',
        recordingStatusCallbackEvent='completed'
    )
    
    print(f"Generated TwiML: {str(response)}")
    return str(response)

@app.route("/process_audio", methods=['POST'])
def process_audio():
    """Process audio from Twilio and return emotion predictions"""
    print("\n=== Processing Audio ===")
    print(f"Headers: {dict(request.headers)}")
    print(f"Form data: {request.form}")
    print(f"Query params: {request.args}")
    print("=====================\n")
    
    try:
        # Get the recording URL
        recording_url = request.values.get('RecordingUrl')
        print(f"Recording URL: {recording_url}")
        
        if not recording_url:
            print("No recording URL received")
            response = VoiceResponse()
            response.say("Could not get the audio recording. Please try speaking again.", voice='Polly.Amy')
            response.redirect('/voice')
            return str(response)
        
        # Extract recording SID from URL
        recording_sid = recording_url.split('/')[-1]
        print(f"Recording SID: {recording_sid}")
        
        # Download the recording
        print("Fetching recording from Twilio")
        recording = client.recordings(recording_sid).fetch()
        print(f"Recording fetched successfully: {recording.uri}")
        
        # Download the actual audio file
        audio_url = f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}/Recordings/{recording_sid}.mp3"
        print(f"Audio URL: {audio_url}")
        
        # Predict emotion
        results = predict_emotion(audio_url)
        
        if results:
            # Format response
            response = VoiceResponse()
            emotion_text = ', '.join([f'{r["emotion"]} with {r["confidence"]:.0%} confidence' for r in results])
            response.say(f"Detected emotions: {emotion_text}", voice='Polly.Amy')
            response.say("Thank you for using our emotion analysis service. Goodbye!", voice='Polly.Amy')
            response.hangup()
            print("Successfully processed audio and returning results")
            return str(response)
        else:
            print("No emotions detected")
            response = VoiceResponse()
            response.say("Could not detect emotions. Please try speaking again.", voice='Polly.Amy')
            response.redirect('/voice')
            return str(response)
            
    except Exception as e:
        print(f"Error processing audio: {str(e)}")
        response = VoiceResponse()
        response.say("An error occurred. Please try again.", voice='Polly.Amy')
        response.redirect('/voice')
        return str(response)

def start_call(to_number):
    """Start a call to the specified number"""
    try:
        print(f"\nStarting call to {to_number}")
        call = client.calls.create(
            to=to_number,
            from_=TWILIO_PHONE_NUMBER,
            url=f"{SERVER_URL}/voice"
        )
        print(f"Call started successfully. SID: {call.sid}")
        return call.sid
    except Exception as e:
        print(f"Error starting call: {str(e)}")
        return None

if __name__ == "__main__":
    print("\nStarting Flask server...")
    # Start Flask server
    app.run(host='0.0.0.0', port=PORT, debug=True) 
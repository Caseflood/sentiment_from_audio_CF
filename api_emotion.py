import os
import numpy as np
import tensorflow as tf
import pandas as pd
from opensmile import Smile, FeatureSet, FeatureLevel
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, jsonify
import tempfile
import soundfile as sf
import io
from pydub import AudioSegment
from dotenv import load_dotenv
import base64
import json
from datetime import datetime
import time

# Load environment variables
load_dotenv()

# Force TensorFlow to use CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Audio parameters from environment variables
SAMPLE_RATE = int(os.getenv('SAMPLE_RATE', 16000))
CHANNELS = int(os.getenv('CHANNELS', 1))
WINDOW_SIZE = 5  # 5 seconds
STRIDE = 5  # 1 second stride

# Model parameters from environment variables
MODEL_PATH = os.getenv('MODEL_PATH', 'emotion_model.keras')
DATASET_PATH = os.getenv('DATASET_PATH', 'emotion_dataset.csv')

# OpenSMILE configuration from environment variables
FEATURE_SET = os.getenv('FEATURE_SET', 'ComParE_2016')
FEATURE_LEVEL = os.getenv('FEATURE_LEVEL', 'Functionals')

app = Flask(__name__)

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

def extract_features(audio_data):
    """Extract features from audio data using OpenSMILE"""
    start_time = time.time()
    print("Starting feature extraction")
    
    # Initialize OpenSMILE
    smile = Smile(
        feature_set=getattr(FeatureSet, FEATURE_SET),
        feature_level=getattr(FeatureLevel, FEATURE_LEVEL),
    )
    
    # Save audio to temporary file
    temp_write_start = time.time()
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        temp_file.write(audio_data)
        temp_file.flush()
        temp_write_time = time.time() - temp_write_start
        
        try:
            # Extract features
            feature_start = time.time()
            features = smile.process_file(temp_file.name)
            features_array = features.values.flatten()
            feature_time = time.time() - feature_start
            
            total_time = time.time() - start_time
            print(f"Feature extraction times: Write={temp_write_time*1000:.2f}ms, Extract={feature_time*1000:.2f}ms, Total={total_time*1000:.2f}ms")
            return features_array.reshape(1, -1)
        finally:
            os.unlink(temp_file.name)

def process_audio_segment(audio_segment):
    """Process a single audio segment and return emotion predictions"""
    try:
        start_time = time.time()
        
        # Convert to WAV format
        wav_start = time.time()
        wav_data = audio_segment.export(format='wav').read()
        wav_time = time.time() - wav_start
        
        # Extract features
        feature_start = time.time()
        features = extract_features(wav_data)
        feature_time = time.time() - feature_start
        
        # Scale features
        scale_start = time.time()
        features_scaled = scaler.transform(features)
        scale_time = time.time() - scale_start
        
        # Make prediction
        predict_start = time.time()
        prediction = model.predict(features_scaled, verbose=0)
        predict_time = time.time() - predict_start
        
        # Process results
        process_start = time.time()
        top_3_indices = np.argsort(prediction[0])[-3:][::-1]
        results = []
        for idx in top_3_indices:
            confidence = float(prediction[0][idx])
            if confidence > 0.05:  # Only include emotions with >5% confidence
                results.append({
                    'emotion': emotions[idx],
                    'confidence': confidence
                })
        process_time = time.time() - process_start
        
        total_time = time.time() - start_time
        print(f"\nSegment processing times:")
        print(f"WAV Convert:    {wav_time*1000:.2f}ms")
        print(f"Feature Extract:{feature_time*1000:.2f}ms")
        print(f"Scale Features: {scale_time*1000:.2f}ms")
        print(f"Prediction:     {predict_time*1000:.2f}ms")
        print(f"Result Process: {process_time*1000:.2f}ms")
        print(f"Total Time:     {total_time*1000:.2f}ms\n")
        
        return results
    except Exception as e:
        print(f"Error processing audio segment: {str(e)}")
        return None

def process_audio_with_sliding_window(audio_data, format='wav'):
    """Process audio using sliding window approach"""
    try:
        # Convert audio data to AudioSegment
        if format == 'wav':
            audio = AudioSegment.from_wav(io.BytesIO(audio_data))
        elif format == 'mp3':
            audio = AudioSegment.from_mp3(io.BytesIO(audio_data))
        else:
            raise ValueError(f"Unsupported audio format: {format}")
        
        # Calculate window size and stride in milliseconds
        window_ms = WINDOW_SIZE * 1000
        stride_ms = STRIDE * 1000
        
        # Process each window
        results = []
        for i in range(0, len(audio) - window_ms + 1, stride_ms):
            # Extract window
            window = audio[i:i + window_ms]
            
            # Process window
            window_results = process_audio_segment(window)
            if window_results:
                results.append({
                    'timestamp': i / 1000,  # Convert to seconds
                    'emotions': window_results
                })
        
        return results
    except Exception as e:
        print(f"Error processing audio with sliding window: {str(e)}")
        return None

@app.route('/analyze', methods=['POST'])
def analyze_audio():
    """API endpoint for emotion analysis"""
    api_start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        # Decode data timing
        decode_start = time.time()
        if 'audio' not in request.files and 'audio_data' not in request.form:
            return jsonify({'error': 'No audio data provided'}), 400
        
        audio_format = request.form.get('format', 'wav')
        
        if 'audio' in request.files:
            audio_file = request.files['audio']
            audio_data = audio_file.read()
        else:
            audio_data = base64.b64decode(request.form['audio_data'])
        decode_time = time.time() - decode_start
        
        # Create directory for saved audio if it doesn't exist
        os.makedirs('saved_audio', exist_ok=True)
        
        # Process audio timing
        process_start = time.time()
        results = process_audio_with_sliding_window(audio_data, format=audio_format)
        process_time = time.time() - process_start
        
        if results:
            # Get the top emotion from the first window (or use "unknown" if none)
            top_emotion = "unknown"
            if results[0]['emotions']:
                top_emotion = results[0]['emotions'][0]['emotion']
            
            # Save the audio file with timestamp and emotion
            filename = f"saved_audio/{timestamp}_{top_emotion}.{audio_format}"
            with open(filename, 'wb') as f:
                f.write(audio_data)
            print(f"Saved audio to {filename}")
            
            total_time = time.time() - api_start_time
            
            # Add timing information to response
            response = {
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'window_size': WINDOW_SIZE,
                'stride': STRIDE,
                'saved_file': filename,
                'results': results,
                'timing': {
                    'decode_time_ms': decode_time * 1000,
                    'process_time_ms': process_time * 1000,
                    'total_time_ms': total_time * 1000
                }
            }
            
            print(f"\nAPI Endpoint Times:")
            print(f"Decode Time:  {decode_time*1000:.2f}ms")
            print(f"Process Time: {process_time*1000:.2f}ms")
            print(f"Total Time:   {total_time*1000:.2f}ms\n")
            
            return jsonify(response)
        else:
            # Save the audio file with "unknown" emotion
            filename = f"saved_audio/{timestamp}_unknown.{audio_format}"
            with open(filename, 'wb') as f:
                f.write(audio_data)
            print(f"Saved audio with unknown emotion to {filename}")
            
            return jsonify({'error': 'Failed to process audio', 'saved_file': filename}), 500
            
    except Exception as e:
        print(f"Error in analyze endpoint: {str(e)}")
        
        # Try to save the audio even if processing failed
        try:
            if 'audio_data' in locals():
                filename = f"saved_audio/{timestamp}_error.{audio_format}"
                with open(filename, 'wb') as f:
                    f.write(audio_data)
                print(f"Error occurred but saved audio to {filename}")
                return jsonify({'error': str(e), 'saved_file': filename}), 500
        except:
            pass
            
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=5000, debug=True) 
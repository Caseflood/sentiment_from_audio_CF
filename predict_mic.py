import os
import numpy as np
import tensorflow as tf
import pandas as pd
from opensmile import Smile, FeatureSet, FeatureLevel
from sklearn.preprocessing import StandardScaler
import sounddevice as sd
import soundfile as sf
import tempfile
import queue
import threading
import time
import sys

# Audio recording parameters
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_DURATION = 5  # 5 second window
SILENCE_THRESHOLD = 0.03
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def extract_features(audio_file):
    # Initialize OpenSMILE with the same configuration used for training
    smile = Smile(
        feature_set=FeatureSet.ComParE_2016,
        feature_level=FeatureLevel.Functionals,
    )
    
    # Extract features
    features = smile.process_file(audio_file)
    features_array = features.values.flatten()
    
    return features_array.reshape(1, -1)  # Reshape for single prediction

def load_model_and_scaler():
    # Load the trained model
    model = tf.keras.models.load_model('emotion_model.keras')
    
    # Load the scaler from the training data
    df = pd.read_csv('emotion_dataset.csv')
    X = df.drop('emotion', axis=1).values
    scaler = StandardScaler()
    scaler.fit(X)
    
    # Get emotion labels
    emotions = sorted(df['emotion'].unique())
    
    return model, scaler, emotions

class AudioProcessor:
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.model, self.scaler, self.emotions = load_model_and_scaler()
        
    def audio_callback(self, indata, frames, time, status):
        """Callback function for audio stream"""
        if status:
            print(status)
        self.audio_queue.put(indata.copy())
    
    def process_audio(self, audio_data):
        """Process audio chunk and predict emotion"""
        # Check if audio is too quiet
        if np.abs(audio_data).mean() < SILENCE_THRESHOLD:
            return None
            
        # Save audio to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            sf.write(temp_file.name, audio_data, SAMPLE_RATE)
            
            try:
                # Extract features
                features = extract_features(temp_file.name)
                
                # Scale features
                features_scaled = self.scaler.transform(features)
                
                # Make prediction
                prediction = self.model.predict(features_scaled, verbose=0)
                predicted_class = np.argmax(prediction[0])
                confidence = prediction[0][predicted_class]
                
                # Get top 3 emotions
                top_3_indices = np.argsort(prediction[0])[-3:][::-1]
                results = []
                for idx in top_3_indices:
                    results.append((self.emotions[idx], float(prediction[0][idx])))
                
                return results
                
            except Exception as e:
                print(f"Error processing audio: {str(e)}")
                return None
            finally:
                # Clean up temp file
                os.unlink(temp_file.name)
    
    def print_emotions(self, results):
        """Print emotion predictions with a simple visualization"""
        if not results:
            return
            
        os.system('clear' if os.name == 'posix' else 'cls')
        print("\nEmotion Analysis:")
        print("=" * 50)
        
        # Print top 3 emotions with bar visualization
        for emotion, confidence in results:
            bar_length = int(confidence * 40)
            bar = "█" * bar_length + "░" * (40 - bar_length)
            print(f"{emotion:10} [{bar}] {confidence:.1%}")
        
        print("=" * 50)
        print("\nPress Ctrl+C to stop recording")
    
    def start_recording(self):
        """Start recording from microphone and predict emotions"""
        try:
            self.is_recording = True
            print("\nStarting emotion recognition...")
            print("Recording from microphone. Speak naturally.")
            print("Press Ctrl+C to stop recording\n")
            
            with sd.InputStream(
                channels=CHANNELS,
                samplerate=SAMPLE_RATE,
                callback=self.audio_callback,
                blocksize=int(SAMPLE_RATE * CHUNK_DURATION)
            ):
                while self.is_recording:
                    try:
                        audio_chunk = self.audio_queue.get(timeout=1)
                        results = self.process_audio(audio_chunk)
                        if results:
                            self.print_emotions(results)
                    except queue.Empty:
                        continue
                    
        except KeyboardInterrupt:
            print("\nStopping emotion recognition...")
        finally:
            self.is_recording = False

def main():
    processor = AudioProcessor()
    processor.start_recording()

if __name__ == '__main__':
    main() 
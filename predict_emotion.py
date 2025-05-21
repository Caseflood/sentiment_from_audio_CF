import os
import numpy as np
import tensorflow as tf
import pandas as pd
from opensmile import Smile, FeatureSet, FeatureLevel
from sklearn.preprocessing import StandardScaler
import joblib
import argparse

# Force TensorFlow to use CPU
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

def predict_emotion(audio_file):
    try:
        # Extract features from audio file
        print(f"Extracting features from {audio_file}...")
        features = extract_features(audio_file)
        
        # Load model and scaler
        print("Loading model...")
        model, scaler, emotions = load_model_and_scaler()
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        print("Predicting emotion...")
        prediction = model.predict(features_scaled)
        predicted_class = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class]
        
        # Get emotion label
        predicted_emotion = emotions[predicted_class]
        
        print("\nPrediction Results:")
        print(f"Predicted Emotion: {predicted_emotion}")
        print(f"Confidence: {confidence:.2%}")
        
        # Print top 3 emotions with their confidences
        print("\nTop 3 Emotions:")
        top_3_indices = np.argsort(prediction[0])[-3:][::-1]
        for idx in top_3_indices:
            print(f"{emotions[idx]}: {prediction[0][idx]:.2%}")
            
    except Exception as e:
        print(f"Error processing {audio_file}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Predict emotion from an audio file')
    parser.add_argument('audio_file', help='Path to the audio file')
    args = parser.parse_args()
    
    if not os.path.exists(args.audio_file):
        print(f"Error: Audio file '{args.audio_file}' not found.")
        return
    
    predict_emotion(args.audio_file)

if __name__ == '__main__':
    main() 
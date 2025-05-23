import os
import pandas as pd
import numpy as np
import opensmile
from tqdm import tqdm

# Initialize OpenSMILE with ComParE_2016 feature set
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.Functionals,
)

def get_emotion_from_filename(filename):
    # CREMA-D filename format: ActorID_Sentence_Emotion_Level.wav
    # Emotions: ANG (Anger), DIS (Disgust), FEA (Fear), HAP (Happy), NEU (Neutral), SAD (Sad)
    emotion_map = {
        'ANG': 'angry',
        'DIS': 'disgust',
        'FEA': 'fearful',
        'HAP': 'happy',
        'NEU': 'neutral',
        'SAD': 'sad'
    }
    parts = filename.split('_')
    if len(parts) >= 3:
        emotion_code = parts[2]
        return emotion_map.get(emotion_code, 'unknown')
    return 'unknown'

def process_audio_files(data_dir):
    features_list = []
    labels = []
    filenames = []
    
    # Get all WAV files in the directory
    wav_files = [f for f in os.listdir(data_dir) if f.endswith('.wav')]
    
    for filename in tqdm(wav_files, desc='Processing audio files'):
        file_path = os.path.join(data_dir, filename)
        
        try:
            # Extract features using OpenSMILE
            features = smile.process_file(file_path)
            features_array = features.values.flatten()
            
            # Get emotion label from filename
            emotion = get_emotion_from_filename(filename)
            
            features_list.append(features_array)
            labels.append(emotion)
            filenames.append(filename)
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    # Convert to numpy arrays
    X = np.array(features_list)
    y = np.array(labels)
    
    return X, y, filenames

def create_dataset():
    data_dir = 'AudioWAV'  # Directory containing CREMA-D WAV files
    
    # Process all audio files
    print("Extracting features from audio files...")
    X, y, filenames = process_audio_files(data_dir)
    
    # Create DataFrame with features
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    
    # Add emotion labels and filenames
    df['emotion'] = y
    df['filename'] = filenames
    
    # Save to CSV
    output_file = 'features_new.csv'
    df.to_csv(output_file, index=False)
    print(f"\nDataset saved to {output_file}")
    print(f"Total samples: {len(df)}")
    print("\nEmotion distribution:")
    print(df['emotion'].value_counts())

if __name__ == '__main__':
    create_dataset() 
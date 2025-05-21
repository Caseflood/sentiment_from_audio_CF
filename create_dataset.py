import os
import pandas as pd
import numpy as np
from opensmile import Smile, FeatureSet, FeatureLevel
from tqdm import tqdm

# Initialize OpenSMILE with ComParE_2016 feature set (which is good for emotion recognition)
smile = Smile(
    feature_set=FeatureSet.ComParE_2016,
    feature_level=FeatureLevel.Functionals,
)

def get_emotion_from_filename(filename):
    # The emotion is encoded in the 3rd part of the filename (02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised)
    emotion_map = {
        '01': 'neutral',
        '02': 'calm',
        '03': 'happy',
        '04': 'sad',
        '05': 'angry',
        '06': 'fearful',
        '07': 'disgust',
        '08': 'surprised'
    }
    parts = filename.split('-')
    emotion_code = parts[2]
    return emotion_map.get(emotion_code, 'unknown')

def process_audio_files(data_dir):
    features_list = []
    labels = []
    
    # Get all actor directories
    actor_dirs = [d for d in os.listdir(data_dir) if d.startswith('Actor_')]
    
    for actor_dir in tqdm(actor_dirs, desc='Processing actors'):
        actor_path = os.path.join(data_dir, actor_dir)
        
        # Process each WAV file in the actor directory
        for filename in os.listdir(actor_path):
            if filename.endswith('.wav'):
                file_path = os.path.join(actor_path, filename)
                
                try:
                    # Extract features using OpenSMILE
                    features = smile.process_file(file_path)
                    features_array = features.values.flatten()
                    
                    # Get emotion label from filename
                    emotion = get_emotion_from_filename(filename)
                    
                    features_list.append(features_array)
                    labels.append(emotion)
                    
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
    
    # Convert to numpy arrays
    X = np.array(features_list)
    y = np.array(labels)
    
    return X, y

def create_dataset():
    data_dir = 'data'
    
    # Process all audio files
    print("Extracting features from audio files...")
    X, y = process_audio_files(data_dir)
    
    # Create DataFrame with features
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    
    # Add emotion labels
    df['emotion'] = y
    
    # Save to CSV
    output_file = 'emotion_dataset.csv'
    df.to_csv(output_file, index=False)
    print(f"\nDataset saved to {output_file}")
    print(f"Total samples: {len(df)}")
    print("\nEmotion distribution:")
    print(df['emotion'].value_counts())

if __name__ == '__main__':
    create_dataset() 
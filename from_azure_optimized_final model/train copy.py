import pandas as pd
import numpy as np
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, optimizers, Input, Model
import matplotlib.pyplot as plt
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

# Set all random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

# Configure GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU is available and configured for memory growth")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("No GPU found, using CPU")

def create_model(input_shape, num_classes):
    """Create a larger model with skip connections using Functional API"""
    inputs = Input(shape=(input_shape,))
    
    # First block
    x = layers.Dense(1536, activation='relu', kernel_regularizer=regularizers.l2(0.001))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    
    # Second block with skip connection
    block_2 = layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    block_2 = layers.BatchNormalization()(block_2)
    block_2 = layers.Dropout(0.4)(block_2)
    
    # Third block with skip connection
    block_3 = layers.Dense(768, activation='relu', kernel_regularizer=regularizers.l2(0.001))(block_2)
    block_3 = layers.BatchNormalization()(block_3)
    block_3 = layers.Dropout(0.3)(block_3)
    # Skip connection from block 1 to block 3
    skip_1_3 = layers.Dense(768, activation='linear')(x)
    block_3 = layers.Add()([block_3, skip_1_3])
    block_3 = layers.Activation('relu')(block_3)
    
    # Fourth block with skip connection
    block_4 = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001))(block_3)
    block_4 = layers.BatchNormalization()(block_4)
    block_4 = layers.Dropout(0.3)(block_4)
    # Skip connection from block 2 to block 4
    skip_2_4 = layers.Dense(512, activation='linear')(block_2)
    block_4 = layers.Add()([block_4, skip_2_4])
    block_4 = layers.Activation('relu')(block_4)
    
    # Fifth block with skip connection
    block_5 = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))(block_4)
    block_5 = layers.BatchNormalization()(block_5)
    block_5 = layers.Dropout(0.2)(block_5)
    # Skip connection from block 3 to block 5
    skip_3_5 = layers.Dense(256, activation='linear')(block_3)
    block_5 = layers.Add()([block_5, skip_3_5])
    block_5 = layers.Activation('relu')(block_5)
    
    # Sixth block with skip connection
    block_6 = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))(block_5)
    block_6 = layers.BatchNormalization()(block_6)
    block_6 = layers.Dropout(0.2)(block_6)
    # Skip connection from block 4 to block 6
    skip_4_6 = layers.Dense(128, activation='linear')(block_4)
    block_6 = layers.Add()([block_6, skip_4_6])
    block_6 = layers.Activation('relu')(block_6)
    
    # Seventh block with skip connection
    block_7 = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001))(block_6)
    block_7 = layers.BatchNormalization()(block_7)
    block_7 = layers.Dropout(0.2)(block_7)
    # Skip connection from block 5 to block 7
    skip_5_7 = layers.Dense(64, activation='linear')(block_5)
    block_7 = layers.Add()([block_7, skip_5_7])
    block_7 = layers.Activation('relu')(block_7)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l2(0.001))(block_7)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Use AdamW optimizer with weight decay
    optimizer = optimizers.AdamW(
        learning_rate=0.01,
        weight_decay=0.001
    )
    
    # Compile with sparse categorical crossentropy
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def load_and_preprocess_data(data_path):
    """Load and preprocess the data with additional steps"""
    print("Loading dataset...")
    df = pd.read_csv(data_path)
    df = df.drop("filename", axis=1)
    # Separate features and labels
    X = df.drop('emotion', axis=1).values

    y = df['emotion'].values
    
    # Encode labels to integers
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split the data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, 
        test_size=0.2,
        stratify=y_encoded,
        random_state=42
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, label_encoder

def main():
    # Initialize wandb
    wandb.init(
        project="sentiment-recognition-CF",
        config={
            "learning_rate": 0.01,
            "epochs": 200,
            "batch_size": 128,
            "architecture": "7-layer-dense-with-skip-connections",
            "optimizer": "AdamW",
            "loss": "sparse_categorical_crossentropy",
            "regularization": "L2(0.001)",
            "dropout_rates": [0.4, 0.4, 0.3, 0.3, 0.2, 0.2, 0.2],
            "layer_sizes": [1536, 1024, 768, 512, 256, 128, 64]
        }
    )
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test, label_encoder = load_and_preprocess_data('features_new.csv')
    
    # Get number of classes from label encoder
    num_classes = len(label_encoder.classes_)
    
    # Log dataset info
    wandb.log({
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "num_features": X_train.shape[1],
        "num_classes": num_classes
    })
    
    # Create model
    input_shape = X_train.shape[1]
    model = create_model(input_shape, num_classes)
    
    # Print model summary
    model.summary()
    
    # Train model with early stopping and reduced learning rate on plateau
    print("\nTraining model...")
    
    history = model.fit(
        X_train, y_train,
        epochs=200,
        batch_size=128,  # Increased batch size for better GPU utilization
        validation_split=0.2,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=20,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_accuracy',
                factor=0.5,
                patience=10,
                min_lr=1e-6
            ),
            WandbMetricsLogger(),
            # Save only the best model based on validation loss
            WandbModelCheckpoint(
                "models/best_model.keras",
                monitor='val_accuracy',
                save_best_only=True,
                mode='min'
            )
        ]
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    test_results = model.evaluate(X_test, y_test, verbose=1)
    metrics = ['loss', 'accuracy']
    for i, metric in enumerate(metrics):
        print(f"{metric}: {test_results[i]:.4f}")
        wandb.log({f"test_{metric}": test_results[i]})
    
    # Save the final model
    print("\nSaving model...")
    model.save('emotion_model.keras')
    wandb.save('emotion_model.keras')
    print("Model saved as 'emotion_model.keras'")
    
    # Clean up old checkpoints
    try:
        for file in os.listdir("models"):
            if file != "best_model.keras":
                os.remove(os.path.join("models", file))
    except Exception as e:
        print(f"Warning: Could not clean up old checkpoints: {e}")
    
    # Close wandb run
    wandb.finish()

if __name__ == "__main__":
    main() 
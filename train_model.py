import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, optimizers
import matplotlib.pyplot as plt
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

# Force TensorFlow to use CPU
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def create_model(input_shape, num_classes):
    """Create a larger, more sophisticated model"""
    model = models.Sequential([
        # Input layer with L2 regularization
        layers.Dense(1024, input_dim=input_shape, 
                    activation='relu',
                    kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        # Second hidden layer
        layers.Dense(768, activation='relu',
                    kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        # Third hidden layer
        layers.Dense(512, activation='relu',
                    kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Fourth hidden layer
        layers.Dense(256, activation='relu',
                    kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Fifth hidden layer
        layers.Dense(128, activation='relu',
                    kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        # Sixth hidden layer
        layers.Dense(64, activation='relu',
                    kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        # Output layer with L2 regularization
        layers.Dense(num_classes, activation='softmax',
                    kernel_regularizer=regularizers.l2(0.01))
    ])
    
    # Use a fixed learning rate instead of a schedule
    optimizer = optimizers.Adam(learning_rate=0.001)
    
    # Compile with additional metrics
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', 
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall(),
                tf.keras.metrics.AUC()]
    )
    
    return model

def load_and_preprocess_data(data_path):
    """Load and preprocess the data with additional steps"""
    print("Loading dataset...")
    df = pd.read_csv(data_path)
    
    # Separate features and labels
    X = df.drop('emotion', axis=1).values
    y = df['emotion'].values
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_onehot = tf.keras.utils.to_categorical(y_encoded)
    
    # Split the data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_onehot, 
        test_size=0.2,
        stratify=y_onehot,
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
            "learning_rate": 0.001,
            "epochs": 200,
            "batch_size": 64,
            "architecture": "6-layer-dense",
            "optimizer": "Adam",
            "loss": "categorical_crossentropy",
            "regularization": "L2(0.01)",
            "dropout_rates": [0.4, 0.4, 0.3, 0.3, 0.2, 0.2],
            "layer_sizes": [1024, 768, 512, 256, 128, 64]
        }
    )
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test, label_encoder = load_and_preprocess_data('emotion_dataset.csv')
    
    # Log dataset info
    wandb.log({
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "num_features": X_train.shape[1],
        "num_classes": y_train.shape[1]
    })
    
    # Create model
    input_shape = X_train.shape[1]
    num_classes = y_train.shape[1]
    model = create_model(input_shape, num_classes)
    
    # Print model summary
    model.summary()
    
    # Train model with early stopping and reduced learning rate on plateau
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        epochs=200,
        batch_size=64,
        validation_split=0.2,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6
            ),
            WandbMetricsLogger(),
            # Save only the best model based on validation loss
            WandbModelCheckpoint(
                "models/best_model.keras",
                monitor='val_loss',
                save_best_only=True,
                mode='min'
            )
        ]
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    test_results = model.evaluate(X_test, y_test, verbose=1)
    metrics = ['loss', 'accuracy', 'precision', 'recall', 'auc']
    for metric, value in zip(metrics, test_results):
        print(f"{metric}: {value:.4f}")
        wandb.log({f"test_{metric}": value})
    
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
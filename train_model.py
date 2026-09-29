"""
================================================================================
Deep Learning-Based Fertilizer Recommendation for Sustainable Crop Production
Script: train_model.py
Description: End-to-End training script for Deep Learning (ANN/MLP) Fertilizer
             Recommendation system. Performs data loading, preprocessing,
             stratified train-test split, model building, training, evaluation,
             and artifact serialization.
================================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Deep Learning / TensorFlow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

# Scikit-Learn Preprocessing & Evaluation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Ensure models directory exists
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
DATASET_PATH = os.path.join(os.path.dirname(__file__), "dataset", "Fertilizer_Prediction.csv")
os.makedirs(MODELS_DIR, exist_ok=True)


def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """
    Loads dataset from CSV, strips column whitespace, standardizes column names,
    and checks for missing values.
    """
    print("=" * 70)
    print("STEP 1: LOADING & EXPLORING DATASET")
    print("=" * 70)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at: {filepath}")
        
    df = pd.read_csv(filepath)
    # Strip any leading/trailing spaces from column headers
    df.columns = df.columns.str.strip()
    
    # Standardize 'Temparature' spelling if present
    if 'Temparature' in df.columns:
        df = df.rename(columns={'Temparature': 'Temperature'})
        
    print(f"[+] Dataset loaded successfully! Dimensions: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"[+] Features: {list(df.columns)}")
    
    # Check for missing values
    missing_count = df.isnull().sum().sum()
    print(f"[+] Total Missing / Null Values: {missing_count}")
    if missing_count > 0:
        df = df.dropna()
        print(f"[!] Dropped missing values. New shape: {df.shape}")
        
    print("\nDataset Summary Statistics:")
    print(df.describe().to_string())
    
    print("\nClass Distribution for Target 'Fertilizer Name':")
    print(df['Fertilizer Name'].value_counts().to_string())
    print("-" * 70)
    
    return df


def build_preprocessors(df: pd.DataFrame):
    """
    Builds and fits Scikit-learn preprocessing pipeline:
    - StandardScaler for numerical environmental & nutrient features
    - OneHotEncoder for categorical features (Soil Type, Crop Type)
    - LabelEncoder for target fertilizer classes
    """
    print("=" * 70)
    print("STEP 2: DATA PREPROCESSING & FEATURE PIPELINE")
    print("=" * 70)
    
    feature_cols = ['Temperature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type', 'Nitrogen', 'Potassium', 'Phosphorous']
    target_col = 'Fertilizer Name'
    
    X = df[feature_cols]
    y = df[target_col]
    
    num_features = ['Temperature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
    cat_features = ['Soil Type', 'Crop Type']
    
    print(f"[+] Numerical Features ({len(num_features)}): {num_features}")
    print(f"[+] Categorical Features ({len(cat_features)}): {cat_features}")
    
    # Preprocessor using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_features)
        ],
        remainder='drop'
    )
    
    # Label Encoder for targets
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"[+] Target Classes ({len(label_encoder.classes_)}): {list(label_encoder.classes_)}")
    
    # Stratified Train/Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
    )
    
    print(f"[+] Train/Test Split: {X_train.shape[0]} Training Samples, {X_test.shape[0]} Test Samples")
    
    # Fit preprocessor on training data only (prevents data leakage)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    
    print(f"[+] Processed Input Feature Dimension: {X_train_proc.shape[1]} neurons")
    print("-" * 70)
    
    return preprocessor, label_encoder, X_train_proc, X_test_proc, y_train, y_test


def build_ann_model(input_dim: int, num_classes: int) -> Sequential:
    """
    Constructs a feed-forward Artificial Neural Network (ANN/MLP)
    with ReLU activations, Dropout regularization, and Softmax classification.
    """
    print("=" * 70)
    print("STEP 3: DEEP LEARNING MODEL ARCHITECTURE")
    print("=" * 70)
    
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(64, activation='relu', name="dense_layer_1"),
        Dropout(0.2, name="dropout_layer"),
        Dense(32, activation='relu', name="dense_layer_2"),
        Dense(num_classes, activation='softmax', name="output_classification_layer")
    ], name="Fertilizer_Recommendation_ANN")
    
    model.compile(
        optimizer=Adam(learning_rate=0.01),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    print("-" * 70)
    return model


def train_and_evaluate(model: Sequential, X_train, y_train, X_test, y_test, label_encoder, preprocessor):
    """
    Trains the neural network, evaluates performance on hold-out test data,
    computes confusion matrix & classification report, and saves evaluation plots.
    """
    print("=" * 70)
    print("STEP 4: MODEL TRAINING & CONVERGENCE")
    print("=" * 70)
    
    EPOCHS = 60
    BATCH_SIZE = 8
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )
    
    print("\n" + "=" * 70)
    print("STEP 5: EVALUATION ON TEST SET")
    print("=" * 70)
    
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    
    print(f"[+] Final Training Accuracy : {train_acc * 100:.2f}% (Loss: {train_loss:.4f})")
    print(f"[+] Final Test Accuracy     : {test_acc * 100:.2f}% (Loss: {test_loss:.4f})")
    
    # Predictions on test data
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Classification Report
    class_names = list(label_encoder.classes_)
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot training curves and confusion matrix
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Accuracy Curve
    axes[0].plot(history.history['accuracy'], label='Train Accuracy', color='#1f77b4', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Val/Test Accuracy', color='#ff7f0e', linewidth=2, linestyle='--')
    axes[0].set_title('Model Accuracy vs Epochs', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend(loc='lower right')
    axes[0].grid(True, linestyle=':', alpha=0.6)
    
    # 2. Loss Curve
    axes[1].plot(history.history['loss'], label='Train Loss', color='#2ca02c', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Val/Test Loss', color='#d62728', linewidth=2, linestyle='--')
    axes[1].set_title('Model Loss vs Epochs', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss (Cross-Entropy)')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, linestyle=':', alpha=0.6)
    
    # 3. Confusion Matrix Heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=axes[2])
    axes[2].set_title('Test Set Confusion Matrix', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Predicted Fertilizer')
    axes[2].set_ylabel('True Fertilizer')
    plt.setp(axes[2].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    plt.tight_layout()
    plot_path = os.path.join(MODELS_DIR, "training_evaluation.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[+] Evaluation plots saved to: {plot_path}")
    
    return history


def save_artifacts(model: Sequential, preprocessor: ColumnTransformer, label_encoder: LabelEncoder):
    """
    Serializes trained model and preprocessing transformers to disk.
    """
    print("\n" + "=" * 70)
    print("STEP 6: SAVING MODEL & PREPROCESSING PIPELINE")
    print("=" * 70)
    
    model_path = os.path.join(MODELS_DIR, "fertilizer_ann_model.keras")
    preprocessor_path = os.path.join(MODELS_DIR, "preprocessor.joblib")
    label_encoder_path = os.path.join(MODELS_DIR, "label_encoder.joblib")
    
    # Save Keras Model
    model.save(model_path)
    print(f"[+] Trained ANN Model saved to: {model_path}")
    
    # Save Scikit-Learn Preprocessors
    joblib.dump(preprocessor, preprocessor_path)
    print(f"[+] Preprocessor pipeline saved to: {preprocessor_path}")
    
    joblib.dump(label_encoder, label_encoder_path)
    print(f"[+] Target Label Encoder saved to: {label_encoder_path}")
    
    print("\n[SUCCESS] All training and artifact saving steps completed successfully!")
    print("=" * 70)


def main():
    # 1. Load Data
    df = load_and_clean_data(DATASET_PATH)
    
    # 2. Preprocess & Split
    preprocessor, label_encoder, X_train, X_test, y_train, y_test = build_preprocessors(df)
    
    # 3. Build Model
    input_dim = X_train.shape[1]
    num_classes = len(label_encoder.classes_)
    model = build_ann_model(input_dim, num_classes)
    
    # 4. Train & Evaluate
    train_and_evaluate(model, X_train, y_train, X_test, y_test, label_encoder, preprocessor)
    
    # 5. Save Artifacts
    save_artifacts(model, preprocessor, label_encoder)


if __name__ == "__main__":
    main()

"""
Preprocessing utilities for FER-2013 face emotion dataset.
"""

import numpy as np
import pandas as pd
import cv2
import os
from typing import Tuple, List
from PIL import Image
import argparse


def load_fer2013_from_csv(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load FER-2013 data from CSV file.
    
    Args:
        csv_path: Path to FER-2013 CSV file
        
    Returns:
        Tuple of (images, labels)
    """
    df = pd.read_csv(csv_path)
    
    images = []
    labels = []
    
    for idx, row in df.iterrows():
        # Parse pixel string
        pixels = np.array(row['pixels'].split(), dtype='uint8')
        
        # Reshape to 48x48
        image = pixels.reshape(48, 48)
        
        images.append(image)
        labels.append(int(row['emotion']))
    
    return np.array(images), np.array(labels)


def load_fer2013_from_folders(data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load FER-2013 data from folder structure.
    
    Args:
        data_dir: Path to folder containing emotion subfolders
        
    Returns:
        Tuple of (images, labels)
    """
    emotion_map = {
        'angry': 0,
        'disgust': 1,
        'fear': 2,
        'happy': 3,
        'sad': 4,
        'surprise': 5,
        'neutral': 6
    }
    
    images = []
    labels = []
    
    for emotion_name, emotion_id in emotion_map.items():
        emotion_dir = os.path.join(data_dir, emotion_name)
        
        if not os.path.exists(emotion_dir):
            continue
        
        for filename in os.listdir(emotion_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                filepath = os.path.join(emotion_dir, filename)
                
                # Load and resize to 48x48
                image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    image = cv2.resize(image, (48, 48))
                    images.append(image)
                    labels.append(emotion_id)
    
    return np.array(images), np.array(labels)


def load_fer2013_data(
    data_dir: str = "data/fer2013",
    cache_dir: str = "data/processed"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load FER-2013 dataset (training and test splits).
    
    Args:
        data_dir: Directory containing FER-2013 data
        cache_dir: Directory to cache preprocessed data
        
    Returns:
        Tuple of (train_images, train_labels, test_images, test_labels)
    """
    # Check for cached data
    cache_file = os.path.join(cache_dir, "fer2013_processed.npz")
    
    if os.path.exists(cache_file):
        print("Loading cached FER-2013 data...")
        data = np.load(cache_file)
        return (
            data['train_images'],
            data['train_labels'],
            data['test_images'],
            data['test_labels']
        )
    
    print("Loading FER-2013 data from source...")
    
    # Try CSV format first
    train_csv = os.path.join(data_dir, "train.csv")
    test_csv = os.path.join(data_dir, "test.csv")
    
    if os.path.exists(train_csv) and os.path.exists(test_csv):
        print("Loading from CSV files...")
        train_images, train_labels = load_fer2013_from_csv(train_csv)
        test_images, test_labels = load_fer2013_from_csv(test_csv)
    
    # Try folder structure
    elif os.path.exists(os.path.join(data_dir, "train")) and os.path.exists(os.path.join(data_dir, "test")):
        print("Loading from folder structure...")
        train_images, train_labels = load_fer2013_from_folders(os.path.join(data_dir, "train"))
        test_images, test_labels = load_fer2013_from_folders(os.path.join(data_dir, "test"))
    
    else:
        raise FileNotFoundError(
            f"FER-2013 data not found in {data_dir}. "
            "Please ensure train.csv/test.csv or train/test folders exist."
        )
    
    # Normalize images to [0, 1]
    train_images = train_images.astype(np.float32) / 255.0
    test_images = test_images.astype(np.float32) / 255.0
    
    # Cache processed data
    os.makedirs(cache_dir, exist_ok=True)
    np.savez(
        cache_file,
        train_images=train_images,
        train_labels=train_labels,
        test_images=test_images,
        test_labels=test_labels
    )
    print(f"Cached preprocessed data to {cache_file}")
    
    print(f"Loaded {len(train_images)} training samples and {len(test_images)} test samples")
    
    return train_images, train_labels, test_images, test_labels


def verify_fer2013_data(data_dir: str = "data/fer2013") -> bool:
    """
    Verify FER-2013 dataset is properly formatted.
    
    Args:
        data_dir: Directory containing FER-2013 data
        
    Returns:
        True if data is valid
    """
    try:
        train_images, train_labels, test_images, test_labels = load_fer2013_data(data_dir)
        
        print("✓ FER-2013 dataset verification:")
        print(f"  Training samples: {len(train_images)}")
        print(f"  Test samples: {len(test_images)}")
        print(f"  Image shape: {train_images[0].shape}")
        print(f"  Number of classes: {len(np.unique(train_labels))}")
        print(f"  Label range: {train_labels.min()} - {train_labels.max()}")
        
        return True
    except Exception as e:
        print(f"✗ Verification failed: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess FER-2013 dataset")
    parser.add_argument("--data-dir", default="data/fer2013", help="FER-2013 data directory")
    parser.add_argument("--cache-dir", default="data/processed", help="Cache directory")
    parser.add_argument("--verify", action="store_true", help="Verify dataset")
    
    args = parser.parse_args()
    
    if args.verify:
        verify_fer2013_data(args.data_dir)
    else:
        print("Preprocessing FER-2013 dataset...")
        train_images, train_labels, test_images, test_labels = load_fer2013_data(
            args.data_dir,
            args.cache_dir
        )
        print("Preprocessing complete!")


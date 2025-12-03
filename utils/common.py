"""
Common utilities for the emotion-aware AI tutor system.
"""

import numpy as np
import os
from typing import Dict, Any, Optional
import json


def ensure_dir(directory: str):
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        directory: Directory path
    """
    os.makedirs(directory, exist_ok=True)


def save_config(config: Dict[str, Any], filepath: str):
    """
    Save configuration dictionary to JSON file.
    
    Args:
        config: Configuration dictionary
        filepath: Path to save config
    """
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)


def load_config(filepath: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        filepath: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def normalize_emotions(emotions: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize emotion probabilities to sum to 1.0.
    
    Args:
        emotions: Dictionary of emotion probabilities
        
    Returns:
        Normalized emotion dictionary
    """
    total = sum(emotions.values())
    if total > 0:
        return {k: v / total for k, v in emotions.items()}
    return emotions


def get_top_emotions(emotions: Dict[str, float], top_k: int = 3) -> Dict[str, float]:
    """
    Get top K emotions by probability.
    
    Args:
        emotions: Dictionary of emotion probabilities
        top_k: Number of top emotions to return
        
    Returns:
        Dictionary of top K emotions
    """
    sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
    return dict(sorted_emotions[:top_k])


def merge_emotions(
    face_emotions: Optional[Dict[str, float]],
    audio_emotions: Optional[Dict[str, float]],
    weights: Tuple[float, float] = (0.6, 0.4)
) -> Dict[str, float]:
    """
    Merge face and audio emotions with weighted combination.
    
    Args:
        face_emotions: Face emotion probabilities
        audio_emotions: Audio emotion probabilities
        weights: Weights for (face, audio)
        
    Returns:
        Merged emotion dictionary
    """
    if not face_emotions and not audio_emotions:
        return {}
    
    if not face_emotions:
        return audio_emotions or {}
    
    if not audio_emotions:
        return face_emotions
    
    # Normalize weights
    total_weight = weights[0] + weights[1]
    face_weight = weights[0] / total_weight
    audio_weight = weights[1] / total_weight
    
    # Get all unique emotions
    all_emotions = set(list(face_emotions.keys()) + list(audio_emotions.keys()))
    
    # Merge with weights
    merged = {}
    for emotion in all_emotions:
        face_prob = face_emotions.get(emotion, 0.0)
        audio_prob = audio_emotions.get(emotion, 0.0)
        merged[emotion] = face_weight * face_prob + audio_weight * audio_prob
    
    return normalize_emotions(merged)


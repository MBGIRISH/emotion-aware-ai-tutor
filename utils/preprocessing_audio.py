"""
Preprocessing utilities for RAVDESS audio emotion dataset.
"""

import numpy as np
import os
from typing import Tuple, List
import argparse
import glob
import soundfile as sf
from scipy import signal
from scipy.fft import dct


def extract_mfcc_simple(audio: np.ndarray, sr: int, n_mfcc: int = 13) -> np.ndarray:
    """
    Extract MFCC features using scipy (no librosa/numba dependency).
    
    Args:
        audio: Audio signal
        sr: Sample rate
        n_mfcc: Number of MFCC coefficients
        
    Returns:
        MFCC features (n_mfcc, time_frames)
    """
    # Parameters
    n_fft = 2048
    hop_length = 512
    n_mels = 40
    
    # Compute power spectrogram
    f, t, S = signal.spectrogram(
        audio,
        fs=sr,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        window='hann'
    )
    
    # Convert to power (magnitude squared)
    S = np.abs(S) ** 2
    
    # Create mel filter bank
    mel_filters = create_mel_filterbank(sr, n_fft, n_mels)
    
    # Apply mel filterbank
    mel_spectrogram = np.dot(mel_filters, S)
    
    # Convert to log scale
    mel_spectrogram = np.log(mel_spectrogram + 1e-10)
    
    # Apply DCT to get MFCC
    mfccs = dct(mel_spectrogram, axis=0, norm='ortho')[:n_mfcc]
    
    return mfccs


def create_mel_filterbank(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    """
    Create mel filterbank.
    
    Args:
        sr: Sample rate
        n_fft: FFT size
        n_mels: Number of mel filters
        
    Returns:
        Mel filterbank matrix
    """
    # Mel scale conversion
    def hz_to_mel(f):
        return 2595 * np.log10(1 + f / 700)
    
    def mel_to_hz(m):
        return 700 * (10 ** (m / 2595) - 1)
    
    # Frequency bins
    fft_freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    
    # Mel scale range
    mel_min = hz_to_mel(0)
    mel_max = hz_to_mel(sr / 2)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    
    # Create filterbank
    filterbank = np.zeros((n_mels, len(fft_freqs)))
    
    for i in range(n_mels):
        # Find frequencies in range
        lower = hz_points[i]
        center = hz_points[i + 1]
        upper = hz_points[i + 2]
        
        # Create triangular filter
        for j, freq in enumerate(fft_freqs):
            if lower <= freq <= center:
                filterbank[i, j] = (freq - lower) / (center - lower)
            elif center < freq <= upper:
                filterbank[i, j] = (upper - freq) / (upper - center)
    
    return filterbank


def load_ravdess_audio_files(data_dir: str) -> Tuple[List[np.ndarray], List[int]]:
    """
    Load RAVDESS audio files and extract MFCC features.
    
    Args:
        data_dir: Directory containing RAVDESS Actor_XX folders
        
    Returns:
        Tuple of (features_list, labels_list)
    """
    # RAVDESS emotion mapping from filename
    # Format: [Modality]-[VocalChannel]-[Emotion]-[EmotionalIntensity]-[Statement]-[Repetition]-[Actor].wav
    # Emotion: 01=Neutral, 02=Calm, 03=Happy, 04=Sad, 05=Angry, 06=Fearful, 07=Disgust, 08=Surprised
    
    features = []
    labels = []
    
    # Map RAVDESS emotion codes to our labels (0-7)
    emotion_map = {
        1: 0,  # Neutral
        2: 1,  # Calm
        3: 2,  # Happy
        4: 3,  # Sad
        5: 4,  # Angry
        6: 5,  # Fearful
        7: 6,  # Disgust
        8: 7   # Surprised
    }
    
    # Find all audio files
    audio_files = glob.glob(os.path.join(data_dir, "Actor_*", "*.wav"))
    
    if not audio_files:
        raise FileNotFoundError(f"No audio files found in {data_dir}")
    
    print(f"Found {len(audio_files)} audio files")
    
    for filepath in audio_files:
        try:
            # Extract emotion from filename
            filename = os.path.basename(filepath)
            parts = filename.split('-')
            
            if len(parts) >= 3:
                emotion_code = int(parts[2])
                
                if emotion_code in emotion_map:
                    # Load audio using soundfile (avoids librosa/numba dependency)
                    audio, sr = sf.read(filepath)
                    
                    # Convert to mono if stereo
                    if len(audio.shape) > 1:
                        audio = np.mean(audio, axis=1)
                    
                    # Resample to 22050 Hz if needed
                    if sr != 22050:
                        num_samples = int(len(audio) * 22050 / sr)
                        audio = signal.resample(audio, num_samples)
                        sr = 22050
                    
                    # Limit to 3 seconds
                    max_samples = 3 * sr
                    if len(audio) > max_samples:
                        audio = audio[:max_samples]
                    
                    # Extract MFCC features using scipy
                    mfccs = extract_mfcc_simple(audio, sr, n_mfcc=13)
                    
                    # Transpose to (time, features)
                    mfccs = mfccs.T
                    
                    # Pad or truncate to fixed length (100 frames ~2-3 seconds)
                    target_length = 100
                    if mfccs.shape[0] < target_length:
                        pad_length = target_length - mfccs.shape[0]
                        mfccs = np.pad(mfccs, ((0, pad_length), (0, 0)), mode='constant')
                    elif mfccs.shape[0] > target_length:
                        mfccs = mfccs[:target_length]
                    
                    # Normalize
                    mfccs = (mfccs - np.mean(mfccs)) / (np.std(mfccs) + 1e-8)
                    
                    features.append(mfccs)
                    labels.append(emotion_map[emotion_code])
        
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            continue
    
    return features, labels


def load_ravdess_data(
    data_dir: str = "data/ravdess",
    cache_dir: str = "data/processed"
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Load RAVDESS dataset with MFCC features.
    
    Args:
        data_dir: Directory containing RAVDESS data
        cache_dir: Directory to cache preprocessed data
        
    Returns:
        Tuple of (features_list, labels_list)
    """
    # Check for cached data
    cache_file = os.path.join(cache_dir, "ravdess_processed.npz")
    
    if os.path.exists(cache_file):
        print("Loading cached RAVDESS data...")
        data = np.load(cache_file, allow_pickle=True)
        return data['features'].tolist(), data['labels'].tolist()
    
    print("Loading RAVDESS data from source...")
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"RAVDESS data directory not found: {data_dir}. "
            "Please ensure RAVDESS dataset is extracted to this location."
        )
    
    features, labels = load_ravdess_audio_files(data_dir)
    
    # Cache processed data
    os.makedirs(cache_dir, exist_ok=True)
    np.savez(
        cache_file,
        features=features,
        labels=labels
    )
    print(f"Cached preprocessed data to {cache_file}")
    
    print(f"Loaded {len(features)} audio samples")
    print(f"Feature shape: {features[0].shape if features else 'N/A'}")
    print(f"Number of classes: {len(np.unique(labels))}")
    
    return features, labels


def verify_ravdess_data(data_dir: str = "data/ravdess") -> bool:
    """
    Verify RAVDESS dataset is properly formatted.
    
    Args:
        data_dir: Directory containing RAVDESS data
        
    Returns:
        True if data is valid
    """
    try:
        features, labels = load_ravdess_data(data_dir)
        
        print("✓ RAVDESS dataset verification:")
        print(f"  Total samples: {len(features)}")
        if features:
            print(f"  Feature shape: {np.array(features[0]).shape}")
        else:
            print(f"  Feature shape: N/A")
        print(f"  Number of classes: {len(np.unique(labels)) if labels else 0}")
        if labels:
            print(f"  Label range: {min(labels)} - {max(labels)}")
        else:
            print(f"  Label range: N/A")
        
        # Check class distribution
        unique, counts = np.unique(labels, return_counts=True)
        print(f"  Class distribution:")
        emotion_names = ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised']
        for label, count in zip(unique, counts):
            print(f"    {emotion_names[label]}: {count}")
        
        return True
    except Exception as e:
        print(f"✗ Verification failed: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess RAVDESS dataset")
    parser.add_argument("--data-dir", default="data/ravdess", help="RAVDESS data directory")
    parser.add_argument("--cache-dir", default="data/processed", help="Cache directory")
    parser.add_argument("--verify", action="store_true", help="Verify dataset")
    
    args = parser.parse_args()
    
    if args.verify:
        verify_ravdess_data(args.data_dir)
    else:
        print("Preprocessing RAVDESS dataset...")
        features, labels = load_ravdess_data(args.data_dir, args.cache_dir)
        print("Preprocessing complete!")


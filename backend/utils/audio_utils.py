"""
Audio processing utilities for emotion detection.
"""

import numpy as np
import librosa
import soundfile as sf
from typing import Tuple, Optional
import io


class AudioProcessor:
    """Audio processing utilities for feature extraction"""
    
    def __init__(
        self,
        sample_rate: int = 22050,
        n_mfcc: int = 13,
        hop_length: int = 512,
        n_fft: int = 2048
    ):
        """
        Initialize audio processor.
        
        Args:
            sample_rate: Target sample rate
            n_mfcc: Number of MFCC coefficients
            hop_length: Hop length for STFT
            n_fft: FFT window size
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.hop_length = hop_length
        self.n_fft = n_fft
    
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        audio, sr = librosa.load(file_path, sr=self.sample_rate)
        return audio, sr
    
    def load_audio_from_bytes(self, audio_bytes: bytes) -> Tuple[np.ndarray, int]:
        """
        Load audio from bytes.
        
        Args:
            audio_bytes: Raw audio bytes
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        audio, sr = sf.read(io.BytesIO(audio_bytes))
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio)
        
        # Resample if needed
        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
        
        return audio, self.sample_rate
    
    def extract_mfcc(
        self,
        audio: np.ndarray,
        n_mfcc: Optional[int] = None
    ) -> np.ndarray:
        """
        Extract MFCC features from audio.
        
        Args:
            audio: Audio signal array
            n_mfcc: Number of MFCC coefficients (uses instance default if None)
            
        Returns:
            MFCC features (n_mfcc, time_frames)
        """
        n_mfcc = n_mfcc or self.n_mfcc
        
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=n_mfcc,
            hop_length=self.hop_length,
            n_fft=self.n_fft
        )
        
        return mfccs
    
    def extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract mel spectrogram from audio.
        
        Args:
            audio: Audio signal array
            
        Returns:
            Mel spectrogram (n_mels, time_frames)
        """
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            n_fft=self.n_fft
        )
        
        # Convert to dB
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_spec_db
    
    def extract_chroma(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract chroma features from audio.
        
        Args:
            audio: Audio signal array
            
        Returns:
            Chroma features (12, time_frames)
        """
        chroma = librosa.feature.chroma(
            y=audio,
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        return chroma
    
    def extract_zero_crossing_rate(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract zero crossing rate from audio.
        
        Args:
            audio: Audio signal array
            
        Returns:
            Zero crossing rate (1, time_frames)
        """
        zcr = librosa.feature.zero_crossing_rate(audio, hop_length=self.hop_length)
        return zcr
    
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio to [-1, 1] range.
        
        Args:
            audio: Audio signal array
            
        Returns:
            Normalized audio
        """
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        return audio
    
    def trim_silence(self, audio: np.ndarray, top_db: int = 20) -> np.ndarray:
        """
        Trim silence from beginning and end of audio.
        
        Args:
            audio: Audio signal array
            top_db: Top dB for silence detection
            
        Returns:
            Trimmed audio
        """
        trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
        return trimmed
    
    def pad_or_truncate(
        self,
        audio: np.ndarray,
        target_length: int,
        mode: str = "constant"
    ) -> np.ndarray:
        """
        Pad or truncate audio to target length.
        
        Args:
            audio: Audio signal array
            target_length: Target length in samples
            mode: Padding mode ('constant', 'edge', etc.)
            
        Returns:
            Padded/truncated audio
        """
        current_length = len(audio)
        
        if current_length < target_length:
            # Pad
            pad_length = target_length - current_length
            audio = np.pad(audio, (0, pad_length), mode=mode)
        elif current_length > target_length:
            # Truncate
            audio = audio[:target_length]
        
        return audio


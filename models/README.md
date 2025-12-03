# Models Directory

This directory stores trained model files for face and audio emotion detection.

## 📁 Files

- `face_emotion_model.pth` - Trained PyTorch model for FER-2013 face emotion recognition
- `audio_emotion_model.pth` - Trained PyTorch model for RAVDESS audio emotion recognition

## 🎯 Model Details

### Face Emotion Model
- **Architecture**: CNN or MobileNetV2 (configurable)
- **Input**: 48x48 grayscale face images
- **Output**: 7 emotion classes (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral)
- **Training**: See `notebooks/train_face_emotion.ipynb`

### Audio Emotion Model
- **Architecture**: LSTM or CNN (configurable)
- **Input**: MFCC features (13-40 coefficients, variable sequence length)
- **Output**: 8 emotion classes (Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised)
- **Training**: See `notebooks/train_audio_emotion.ipynb`

## 📥 Downloading Pre-trained Models

If you have pre-trained models, place them here with the exact filenames above.

## 🔄 Training New Models

1. Ensure datasets are in `data/fer2013/` and `data/ravdess/`
2. Run preprocessing: `python utils/preprocessing_face.py` and `python utils/preprocessing_audio.py`
3. Train models using the Jupyter notebooks in `notebooks/`
4. Models will be automatically saved to this directory

## ⚠️ Note

Models are not included in the repository due to size. You must train them or download pre-trained versions separately.


# Project Structure

Complete file structure of the emotion-aware AI tutor system.

```
emotion-aware-ai-tutor/
├── README.md                          # Main documentation
├── QUICKSTART.md                      # Quick start guide
├── PROJECT_STRUCTURE.md               # This file
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules
├── .env.example                       # Environment variables template
│
├── data/
│   ├── fer2013/                       # FER-2013 dataset (user uploads here)
│   ├── ravdess/                       # RAVDESS dataset (user uploads here)
│   ├── processed/                     # Preprocessed data cache
│   └── instructions.md                 # Dataset upload instructions
│
├── models/
│   ├── face_emotion_model.pth         # Trained face emotion model (after training)
│   ├── audio_emotion_model.pth        # Trained audio emotion model (after training)
│   └── README.md                      # Model documentation
│
├── backend/
│   ├── api.py                         # FastAPI server
│   ├── inference_face.py              # Face emotion inference
│   ├── inference_audio.py             # Audio emotion inference
│   ├── engagement.py                  # Engagement & confusion tracking
│   ├── tutor.py                       # LLM adaptive tutor
│   └── utils/
│       ├── __init__.py
│       ├── face_detector.py           # MediaPipe face detection
│       ├── audio_utils.py             # Audio processing utilities
│       └── logger.py                   # Logging utilities
│
├── app/
│   ├── streamlit_app.py               # Main Streamlit dashboard
│   └── components/
│       ├── __init__.py
│       ├── emotion_meter.py            # Face emotion visualization
│       ├── voice_gauge.py              # Audio emotion visualization
│       ├── engagement_bar.py           # Engagement score display
│       └── tutor_chatbox.py            # Tutor chat interface
│
├── notebooks/
│   ├── train_face_emotion.ipynb       # Train FER-2013 model
│   ├── train_audio_emotion.ipynb      # Train RAVDESS model (TODO: create)
│   ├── evaluate_models.ipynb          # Model evaluation (TODO: create)
│   └── demo_predictions.ipynb         # Demo inference (TODO: create)
│
└── utils/
    ├── __init__.py
    ├── preprocessing_face.py          # FER-2013 preprocessing
    ├── preprocessing_audio.py         # RAVDESS preprocessing
    └── common.py                      # Common utilities
```

## File Descriptions

### Core Backend Files

- **backend/api.py**: FastAPI REST API server for emotion inference
- **backend/inference_face.py**: Real-time face emotion detection using trained model
- **backend/inference_audio.py**: Real-time audio emotion detection using trained model
- **backend/engagement.py**: Computes engagement score from blink rate, gaze, and emotions
- **backend/tutor.py**: LLM-powered adaptive tutoring responses

### Frontend Files

- **app/streamlit_app.py**: Main dashboard application
- **app/components/**: Reusable Streamlit visualization components

### Training & Evaluation

- **notebooks/train_face_emotion.ipynb**: Complete training pipeline for FER-2013
- **notebooks/train_audio_emotion.ipynb**: Training pipeline for RAVDESS (to be created)
- **notebooks/evaluate_models.ipynb**: Model evaluation and metrics (to be created)
- **notebooks/demo_predictions.ipynb**: Demo inference examples (to be created)

### Utilities

- **utils/preprocessing_face.py**: Load and preprocess FER-2013 dataset
- **utils/preprocessing_audio.py**: Load and preprocess RAVDESS dataset
- **utils/common.py**: Shared utility functions

## Next Steps

1. Create remaining notebooks (train_audio_emotion, evaluate_models, demo_predictions)
2. Add unit tests
3. Add Docker support
4. Add CI/CD pipeline


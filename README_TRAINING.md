# 🚀 Training Guide - Complete Instructions

## ✅ Setup Complete

- ✅ Python 3.13 environment created (`venv313/`)
- ✅ All dependencies installed (including MediaPipe support)
- ✅ Jupyter kernel configured: **"Python (emotion-ai-tutor-313)"**
- ✅ Notebooks fixed with correct model architecture
- ✅ Training script ready: `train_models_fixed.py`

## 🎯 How to Train Models

### Option 1: Automated Training Script (Recommended)

```bash
cd /Users/mbgirish/emotion-aware-ai-tutor
source venv313/bin/activate
python train_models_fixed.py
```

This will train both models automatically. Estimated time: 35-50 minutes.

### Option 2: Jupyter Notebooks (Interactive)

**Train Face Model:**
```bash
source venv313/bin/activate
jupyter notebook notebooks/train_face_emotion.ipynb
```
- Select kernel: **"Python (emotion-ai-tutor-313)"**
- Run all cells
- Model saves to `models/face_emotion_model.pth`

**Train Audio Model:**
```bash
jupyter notebook notebooks/train_audio_emotion.ipynb
```
- Select kernel: **"Python (emotion-ai-tutor-313)"**
- Run all cells
- Model saves to `models/audio_emotion_model.pth`

## 📊 Monitor Training

```bash
# Watch training progress
tail -f training_log.txt

# Check if models are created
ls -lh models/*.pth

# Check if training is running
ps aux | grep train_models
```

## ✅ After Training

Once both models are trained:

1. **Start FastAPI Backend:**
   ```bash
   source venv313/bin/activate
   cd backend
   uvicorn api:app --reload
   ```

2. **Start Streamlit Dashboard:**
   ```bash
   source venv313/bin/activate
   streamlit run app/streamlit_app.py
   ```

3. **Open browser:** `http://localhost:8501`

## 🔧 Fixed Issues

- ✅ Model architecture corrected (128*3*3 instead of 128*6*6)
- ✅ Dataset loading fixed
- ✅ MediaPipe made optional (OpenCV fallback)
- ✅ Python 3.13 environment with all dependencies

## 📝 Notes

- Training takes 30-60 minutes total
- Models will be ~10-50MB each
- Use Python 3.13 environment (`venv313/`) for full MediaPipe support
- Original `venv/` works for training but MediaPipe unavailable

---

**Everything is ready! Start training now!** 🚀


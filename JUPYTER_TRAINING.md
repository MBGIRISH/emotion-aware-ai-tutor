# 🚀 How to Train Using Jupyter

## ✅ Quick Start

### Option 1: JupyterLab (Recommended)
```bash
cd /Users/mbgirish/emotion-aware-ai-tutor
source venv313/bin/activate
jupyter lab notebooks/train_face_emotion.ipynb
```

### Option 2: Classic Jupyter Notebook
```bash
cd /Users/mbgirish/emotion-aware-ai-tutor
source venv313/bin/activate
jupyter notebook notebooks/train_face_emotion.ipynb
```

## 📝 Important Notes

1. **Use `venv313` environment** (Python 3.13) for full MediaPipe support
2. **Select the correct kernel** when notebook opens:
   - Choose: **"Python (emotion-ai-tutor-313)"**
3. **Run all cells** to train the model

## 🎯 Training Both Models

**Face Model:**
```bash
jupyter lab notebooks/train_face_emotion.ipynb
```

**Audio Model:**
```bash
jupyter lab notebooks/train_audio_emotion.ipynb
```

## ⚠️ Note

**Training is already running in the background!** 

Check status with:
```bash
./check_training.sh
```

Or monitor with:
```bash
tail -f training_full.log
```

You don't need to start training again unless you want to use Jupyter interactively.

---

**Quick Command:**
```bash
source venv313/bin/activate && jupyter lab notebooks/train_face_emotion.ipynb
```


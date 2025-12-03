# Quick Start Guide

## 🚀 Getting Started in 5 Minutes

### 1. Install Dependencies

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download Datasets

**FER-2013:**
- Download from: https://www.kaggle.com/datasets/msambare/fer2013
- Extract to: `data/fer2013/`

**RAVDESS:**
- Download from: https://zenodo.org/record/1188976
- Extract to: `data/ravdess/`

### 3. Preprocess Data

```bash
python utils/preprocessing_face.py
python utils/preprocessing_audio.py
```

### 4. Train Models

**Option A: Using Jupyter Notebooks (Recommended)**
```bash
jupyter notebook notebooks/train_face_emotion.ipynb
jupyter notebook notebooks/train_audio_emotion.ipynb
```

**Option B: Using Python Scripts**
```python
# Train face model
exec(open('notebooks/train_face_emotion.py').read())

# Train audio model
exec(open('notebooks/train_audio_emotion.py').read())
```

### 5. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your API keys
```

### 6. Start the System

**Terminal 1: Start FastAPI Backend**
```bash
cd backend
uvicorn api:app --reload
```

**Terminal 2: Start Streamlit Dashboard**
```bash
streamlit run app/streamlit_app.py
```

### 7. Access Dashboard

Open your browser to: `http://localhost:8501`

## 📝 Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check [data/instructions.md](data/instructions.md) for dataset setup
- Explore the notebooks for training and evaluation

## 🆘 Troubleshooting

**Models not found?**
- Ensure you've trained the models first (step 4)
- Check that models are saved to `models/` directory

**API connection error?**
- Verify FastAPI backend is running on port 8000
- Check API_URL in Streamlit sidebar

**Webcam not working?**
- Check camera permissions
- Try different camera index (0, 1, 2)

**Dataset errors?**
- Verify datasets are in correct locations
- Run verification: `python utils/preprocessing_face.py --verify`


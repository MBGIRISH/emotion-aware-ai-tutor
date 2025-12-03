# Dataset Upload Instructions

This document explains how to prepare and upload the required datasets for training the emotion-aware AI tutor system.

## 📦 Required Datasets

### 1. FER-2013 (Facial Expression Recognition)

**Purpose:** Train the face emotion detection model

**Download:**
- Kaggle: https://www.kaggle.com/datasets/msambare/fer2013
- Alternative: https://www.kaggle.com/datasets/deadskull7/fer2013

**Upload Location:** `data/fer2013/`

**Accepted Formats:**

**Option A: CSV Format (Recommended)**
```
data/fer2013/
  ├── train.csv
  └── test.csv
```

Each CSV should have columns:
- `emotion`: Integer label (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral)
- `pixels`: Space-separated pixel values (48x48 grayscale image flattened)
- `Usage`: "Training" or "PrivateTest" or "PublicTest"

**Option B: Folder Structure**
```
data/fer2013/
  ├── train/
  │   ├── angry/
  │   ├── disgust/
  │   ├── fear/
  │   ├── happy/
  │   ├── sad/
  │   ├── surprise/
  │   └── neutral/
  └── test/
      ├── angry/
      ├── disgust/
      ├── fear/
      ├── happy/
      ├── sad/
      ├── surprise/
      └── neutral/
```

**Verification:**
After uploading, run:
```bash
python utils/preprocessing_face.py --verify
```

---

### 2. RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)

**Purpose:** Train the voice emotion detection model

**Download:**
- Official: https://zenodo.org/record/1188976
- Direct link: https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip

**Upload Location:** `data/ravdess/`

**Expected Structure:**
```
data/ravdess/
  ├── Actor_01/
  │   ├── 03-01-01-01-01-01-01.wav
  │   ├── 03-01-01-01-01-02-01.wav
  │   └── ...
  ├── Actor_02/
  ├── Actor_03/
  └── ...
```

**File Naming Convention:**
RAVDESS files follow: `[Modality]-[VocalChannel]-[Emotion]-[EmotionalIntensity]-[Statement]-[Repetition]-[Actor].wav`

- **Modality**: 03 = Audio-only
- **Emotion**: 01=Neutral, 02=Calm, 03=Happy, 04=Sad, 05=Angry, 06=Fearful, 07=Disgust, 08=Surprised
- **Actor**: 01-24 (12 male, 12 female)

**Verification:**
After uploading, run:
```bash
python utils/preprocessing_audio.py --verify
```

---

## ✅ Pre-Upload Checklist

- [ ] FER-2013 dataset downloaded
- [ ] FER-2013 placed in `data/fer2013/` (CSV or folder format)
- [ ] RAVDESS dataset downloaded and extracted
- [ ] RAVDESS placed in `data/ravdess/` with Actor_XX folders
- [ ] Both datasets verified using preprocessing scripts

## 🔄 After Upload

1. **Run Preprocessing:**
   ```bash
   python utils/preprocessing_face.py
   python utils/preprocessing_audio.py
   ```

2. **Verify Processed Data:**
   - Check `data/processed/` directory for preprocessed files
   - Ensure no errors in preprocessing logs

3. **Train Models:**
   - Open `notebooks/train_face_emotion.ipynb`
   - Open `notebooks/train_audio_emotion.ipynb`
   - Run all cells to train models

## 📊 Dataset Statistics

**FER-2013:**
- Training: ~28,709 images
- Testing: ~3,589 images
- Classes: 7 emotions
- Image size: 48x48 grayscale

**RAVDESS:**
- Total files: ~1,440 audio files
- Actors: 24 (12 male, 12 female)
- Emotions: 8 classes
- Duration: ~2-3 seconds per file
- Sample rate: 48kHz

## ⚠️ Important Notes

1. **Both training and testing data MUST be uploaded** for proper model evaluation
2. **Do not modify dataset structure** after preprocessing
3. **Ensure sufficient disk space** (~500MB for FER-2013, ~200MB for RAVDESS)
4. **Keep original datasets** - preprocessing creates cached versions but originals are needed for retraining

## 🆘 Troubleshooting

**"Dataset not found" error:**
- Verify paths match exactly: `data/fer2013/` and `data/ravdess/`
- Check file permissions
- Ensure datasets are extracted (not zip files)

**"Invalid format" error:**
- For FER-2013: Check CSV columns or folder structure
- For RAVDESS: Verify Actor_XX folder naming and .wav file format

**"Insufficient data" error:**
- Ensure complete datasets are uploaded
- Check for missing Actor folders in RAVDESS
- Verify train/test split exists for FER-2013


# 📥 Dataset Upload Guide - Step by Step

This guide shows you exactly how to download and upload the datasets.

## 🎯 Quick Summary

1. **FER-2013**: Download from Kaggle → Extract → Place in `data/fer2013/`
2. **RAVDESS**: Download from Zenodo → Extract → Place in `data/ravdess/`

---

## 📊 Step 1: Download FER-2013 Dataset

### Option A: Using Kaggle CLI (Recommended)

```bash
# Install Kaggle CLI if you haven't
pip install kaggle

# Set up Kaggle credentials (get API token from https://www.kaggle.com/settings)
# Place kaggle.json in ~/.kaggle/ directory

# Download FER-2013 dataset
kaggle datasets download -d msambare/fer2013 -p data/fer2013/

# Extract the zip file
cd data/fer2013/
unzip fer2013.zip
cd ../..
```

### Option B: Manual Download

1. **Go to Kaggle**: https://www.kaggle.com/datasets/msambare/fer2013
2. **Click "Download"** (requires Kaggle account)
3. **Extract the zip file**
4. **Copy files to project**:

```bash
# If you downloaded to Downloads folder
cd ~/Downloads
unzip fer2013.zip

# Copy to project (choose the format you have)
# For CSV format:
cp train.csv test.csv /Users/mbgirish/emotion-aware-ai-tutor/data/fer2013/

# OR for folder format:
cp -r train test /Users/mbgirish/emotion-aware-ai-tutor/data/fer2013/
```

### Verify FER-2013 Upload

```bash
# Check if files are in place
ls -la data/fer2013/

# Should see either:
# - train.csv and test.csv (CSV format)
# OR
# - train/ and test/ folders (folder format)

# Verify the dataset
python utils/preprocessing_face.py --verify
```

---

## 🎤 Step 2: Download RAVDESS Dataset

### Using curl (macOS/Linux)

```bash
# Download RAVDESS dataset (curl works on macOS by default)
cd data/ravdess/
curl -L -o Audio_Speech_Actors_01-24.zip https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip

# Extract
unzip Audio_Speech_Actors_01-24.zip

# Clean up zip file
rm Audio_Speech_Actors_01-24.zip

# The extracted folder should contain Actor_01, Actor_02, etc.
# Move contents to current directory if needed
# If it extracted to a subfolder:
# mv Audio_Speech_Actors_01-24/* .
# rmdir Audio_Speech_Actors_01-24

cd ../..
```

### Manual Download

1. **Go to Zenodo**: https://zenodo.org/record/1188976
2. **Click "Download"** on "Audio_Speech_Actors_01-24.zip"
3. **Extract the zip file**
4. **Copy to project**:

```bash
# If downloaded to Downloads
cd ~/Downloads
unzip Audio_Speech_Actors_01-24.zip

# Copy to project
cp -r Audio_Speech_Actors_01-24/* /Users/mbgirish/emotion-aware-ai-tutor/data/ravdess/
```

### Verify RAVDESS Upload

```bash
# Check structure
ls data/ravdess/

# Should see:
# Actor_01/
# Actor_02/
# ...
# Actor_24/

# Check one actor folder
ls data/ravdess/Actor_01/ | head -5
# Should see .wav files like: 03-01-01-01-01-01-01.wav

# Verify the dataset
python utils/preprocessing_audio.py --verify
```

---

## ✅ Step 3: Verify Everything

Run these commands to verify both datasets:

```bash
# Verify FER-2013
python utils/preprocessing_face.py --verify

# Verify RAVDESS
python utils/preprocessing_audio.py --verify
```

You should see output like:
```
✓ FER-2013 dataset verification:
  Training samples: 28709
  Test samples: 3589
  ...

✓ RAVDESS dataset verification:
  Total samples: 1440
  Feature shape: (100, 13)
  ...
```

---

## 🔧 Step 4: Preprocess Data

Once datasets are uploaded, preprocess them:

```bash
# Preprocess FER-2013
python utils/preprocessing_face.py

# Preprocess RAVDESS
python utils/preprocessing_audio.py
```

This will create cached preprocessed files in `data/processed/` for faster loading during training.

---

## 📁 Final Directory Structure

After uploading, your `data/` directory should look like:

```
data/
├── fer2013/
│   ├── train.csv          # OR train/ folder
│   └── test.csv           # OR test/ folder
├── ravdess/
│   ├── Actor_01/
│   │   ├── 03-01-01-01-01-01-01.wav
│   │   └── ...
│   ├── Actor_02/
│   └── ...
└── processed/             # Created after preprocessing
    ├── fer2013_processed.npz
    └── ravdess_processed.npz
```

---

## 🆘 Troubleshooting

### "Dataset not found" error

```bash
# Check if directories exist
ls -la data/fer2013/
ls -la data/ravdess/

# Check file permissions
chmod -R 755 data/fer2013/
chmod -R 755 data/ravdess/
```

### "No files found" error

- **FER-2013**: Make sure `train.csv` and `test.csv` are directly in `data/fer2013/` (not in a subfolder)
- **RAVDESS**: Make sure `Actor_01/`, `Actor_02/`, etc. are directly in `data/ravdess/` (not in a subfolder)

### Download issues

- **Kaggle**: You need a Kaggle account and API token. Get it from https://www.kaggle.com/settings
- **Zenodo**: The download is free, no account needed, but it's a large file (~200MB)

### Disk space

- FER-2013: ~500MB
- RAVDESS: ~200MB
- Processed cache: ~100MB

Check available space:
```bash
df -h .
```

---

## 🚀 Next Steps

After datasets are uploaded and verified:

1. ✅ Preprocess data (Step 4 above)
2. ✅ Train face emotion model: `jupyter notebook notebooks/train_face_emotion.ipynb`
3. ✅ Train audio emotion model: `jupyter notebook notebooks/train_audio_emotion.ipynb`
4. ✅ Start the system and test!


#!/bin/bash
# Script to download and set up datasets

echo "📥 Downloading Datasets for Emotion-Aware AI Tutor"
echo "=================================================="

# Create directories
mkdir -p data/fer2013 data/ravdess

echo ""
echo "1️⃣  FER-2013 Dataset"
echo "   Please download manually from: https://www.kaggle.com/datasets/msambare/fer2013"
echo "   Then extract to: data/fer2013/"
echo ""
echo "   OR use Kaggle CLI:"
echo "   kaggle datasets download -d msambare/fer2013 -p data/fer2013/"
echo "   cd data/fer2013 && unzip fer2013.zip && cd ../.."

echo ""
echo "2️⃣  RAVDESS Dataset"
read -p "   Download RAVDESS now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "   Downloading RAVDESS..."
    cd data/ravdess/
    curl -L -o Audio_Speech_Actors_01-24.zip https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip
    echo "   Extracting..."
    unzip -q Audio_Speech_Actors_01-24.zip
    # Move contents if extracted to subfolder
    if [ -d "Audio_Speech_Actors_01-24" ]; then
        mv Audio_Speech_Actors_01-24/* .
        rmdir Audio_Speech_Actors_01-24
    fi
    rm Audio_Speech_Actors_01-24.zip
    cd ../..
    echo "   ✅ RAVDESS downloaded and extracted!"
else
    echo "   Skipped. Download manually from: https://zenodo.org/record/1188976"
    echo "   Extract to: data/ravdess/"
fi

echo ""
echo "3️⃣  Verification"
read -p "   Verify datasets now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "   Verifying FER-2013..."
    python utils/preprocessing_face.py --verify
    echo ""
    echo "   Verifying RAVDESS..."
    python utils/preprocessing_audio.py --verify
fi

echo ""
echo "✅ Setup complete! Next steps:"
echo "   1. Run preprocessing: python utils/preprocessing_face.py && python utils/preprocessing_audio.py"
echo "   2. Train models using Jupyter notebooks"

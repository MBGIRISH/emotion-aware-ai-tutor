#!/bin/bash
# Script to download FER-2013 dataset

echo "📥 Downloading FER-2013 Dataset"
echo "================================"
echo ""

# Create directory
mkdir -p data/fer2013

# Check if Kaggle CLI is installed
if command -v kaggle &> /dev/null; then
    echo "✅ Kaggle CLI found!"
    echo ""
    echo "Downloading FER-2013 dataset..."
    cd data/fer2013/
    kaggle datasets download -d msambare/fer2013
    echo ""
    echo "Extracting..."
    unzip -q fer2013.zip
    rm fer2013.zip
    cd ../..
    echo "✅ FER-2013 downloaded and extracted!"
else
    echo "❌ Kaggle CLI not found."
    echo ""
    echo "You have two options:"
    echo ""
    echo "Option 1: Install Kaggle CLI"
    echo "  brew install pipx"
    echo "  pipx install kaggle"
    echo "  # Then get your API token from: https://www.kaggle.com/settings"
    echo "  # Place kaggle.json in ~/.kaggle/ directory"
    echo ""
    echo "Option 2: Manual Download (Easier)"
    echo "  1. Go to: https://www.kaggle.com/datasets/msambare/fer2013"
    echo "  2. Click 'Download' button (requires Kaggle account - free to sign up)"
    echo "  3. Extract the zip file"
    echo "  4. Copy train.csv and test.csv to: data/fer2013/"
    echo ""
    echo "After downloading, verify with:"
    echo "  python3 utils/preprocessing_face.py --verify"
fi


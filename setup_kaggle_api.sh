#!/bin/bash
# Script to set up Kaggle API token

echo "🔑 Kaggle API Token Setup"
echo "========================="
echo ""

# Check if kaggle.json exists in Downloads
if [ -f ~/Downloads/kaggle.json ]; then
    echo "✅ Found kaggle.json in Downloads folder"
    echo ""
    echo "Setting up Kaggle API..."
    
    # Create .kaggle directory
    mkdir -p ~/.kaggle
    
    # Copy kaggle.json
    cp ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
    
    # Set correct permissions (required by Kaggle)
    chmod 600 ~/.kaggle/kaggle.json
    
    echo "✅ Kaggle API token configured!"
    echo ""
    echo "You can now use:"
    echo "  kaggle datasets download -d msambare/fer2013 -p data/fer2013/"
    echo ""
    
    # Test if kaggle CLI is installed
    if command -v kaggle &> /dev/null; then
        echo "Testing Kaggle API..."
        kaggle datasets list --max-size 1000000 | head -5
        echo ""
        echo "✅ Kaggle CLI is working!"
    else
        echo "⚠️  Kaggle CLI not installed yet."
        echo "   Install with: pipx install kaggle"
    fi
else
    echo "❌ kaggle.json not found in Downloads folder"
    echo ""
    echo "📥 To get your API token:"
    echo "   1. Go to: https://www.kaggle.com/settings"
    echo "   2. Scroll to 'API' section"
    echo "   3. Click 'Generate New Token' button"
    echo "   4. A file named 'kaggle.json' will download"
    echo "   5. Run this script again, or manually:"
    echo ""
    echo "   mkdir -p ~/.kaggle"
    echo "   cp ~/Downloads/kaggle.json ~/.kaggle/kaggle.json"
    echo "   chmod 600 ~/.kaggle/kaggle.json"
    echo ""
fi


#!/bin/bash
# Quick script to check training progress

cd /Users/mbgirish/emotion-aware-ai-tutor

echo "📊 Training Progress Check"
echo "========================"
echo ""

# Check model files
if [ -f models/face_emotion_model.pth ] && [ -f models/audio_emotion_model.pth ]; then
    echo "🎉 TRAINING COMPLETE! 🎉"
    echo ""
    ls -lh models/*.pth
    exit 0
fi

# Check log file
if [ ! -f training_full.log ]; then
    echo "⚠️  Training log not found - training may not have started"
    exit 1
fi

# Count lines in log
LOG_LINES=$(wc -l < training_full.log)
echo "Log file: $LOG_LINES lines"
echo ""

# Check for face model progress
if grep -q "TRAINING FACE EMOTION MODEL" training_full.log; then
    FACE_EPOCH=$(grep "Epoch \[" training_full.log | grep "FACE" -A 1 | tail -1 | grep -o "Epoch \[[0-9]*/30\]" | grep -o "[0-9]*" | head -1)
    if [ ! -z "$FACE_EPOCH" ]; then
        FACE_PERCENT=$((FACE_EPOCH * 100 / 30))
        echo "✅ Face Model: Epoch $FACE_EPOCH/30 ($FACE_PERCENT%)"
    else
        echo "⏳ Face Model: Starting..."
    fi
    
    if grep -q "Saved to models/face_emotion_model.pth" training_full.log; then
        echo "   ✅ Face model saved!"
    fi
else
    echo "⏳ Face Model: Not started"
fi

echo ""

# Check for audio model progress
if grep -q "TRAINING AUDIO EMOTION MODEL" training_full.log; then
    AUDIO_EPOCH=$(grep "Epoch \[" training_full.log | grep "AUDIO" -A 1 | tail -1 | grep -o "Epoch \[[0-9]*/50\]" | grep -o "[0-9]*" | head -1)
    if [ ! -z "$AUDIO_EPOCH" ]; then
        AUDIO_PERCENT=$((AUDIO_EPOCH * 100 / 50))
        echo "✅ Audio Model: Epoch $AUDIO_EPOCH/50 ($AUDIO_PERCENT%)"
    else
        echo "⏳ Audio Model: Starting..."
    fi
    
    if grep -q "Saved to models/audio_emotion_model.pth" training_full.log; then
        echo "   ✅ Audio model saved!"
    fi
else
    echo "⏳ Audio Model: Not started"
fi

echo ""
echo "Latest output:"
tail -5 training_full.log


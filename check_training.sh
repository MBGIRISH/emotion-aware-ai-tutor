#!/bin/bash
# Quick training status checker

echo "📊 Training Status"
echo "================="
echo ""

# Check if process is running
if ps aux | grep -q "[p]ython train_models"; then
    PROC_INFO=$(ps aux | grep "[p]ython train_models" | grep -v grep | head -1)
    CPU=$(echo $PROC_INFO | awk '{print $3}')
    TIME=$(echo $PROC_INFO | awk '{print $10}')
    echo "✅ Training Process: RUNNING"
    echo "   CPU: ${CPU}%"
    echo "   Runtime: ${TIME}"
else
    echo "⏳ Training Process: Not running"
fi

echo ""

# Check models
if [ -f models/face_emotion_model.pth ]; then
    echo "✅ Face Model: COMPLETE"
    ls -lh models/face_emotion_model.pth
else
    echo "⏳ Face Model: Not created yet"
fi

echo ""

if [ -f models/audio_emotion_model.pth ]; then
    echo "✅ Audio Model: COMPLETE"
    ls -lh models/audio_emotion_model.pth
else
    echo "⏳ Audio Model: Not created yet"
fi

echo ""

# Check log
if [ -f training_full.log ]; then
    LINES=$(wc -l < training_full.log)
    echo "📝 Log file: $LINES lines"
    if [ $LINES -gt 0 ]; then
        echo ""
        echo "Latest output:"
        tail -3 training_full.log
    fi
else
    echo "📝 Log file: Not found"
fi

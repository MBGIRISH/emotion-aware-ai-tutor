#!/usr/bin/env python3
"""Quick test to verify project is functional"""

import sys
import os
sys.path.append('.')

print("🧪 Testing Emotion-Aware AI Tutor Project")
print("=" * 50)

# Test 1: Data loading
print("\n1. Testing data loading...")
try:
    from utils.preprocessing_face import load_fer2013_data
    train_data, train_labels, test_data, test_labels = load_fer2013_data()
    print(f"   ✅ FER-2013: {len(train_data)} train, {len(test_data)} test")
except Exception as e:
    print(f"   ❌ FER-2013 failed: {e}")

try:
    from utils.preprocessing_audio import load_ravdess_data
    X, y = load_ravdess_data()
    print(f"   ✅ RAVDESS: {len(X)} samples")
except Exception as e:
    print(f"   ❌ RAVDESS failed: {e}")

# Test 2: Backend imports
print("\n2. Testing backend modules...")
modules = [
    ('backend.inference_face', 'FaceEmotionInference'),
    ('backend.inference_audio', 'AudioEmotionInference'),
    ('backend.engagement', 'EngagementTracker'),
    ('backend.tutor', 'AdaptiveTutor'),
    ('backend.utils.face_detector', 'FaceDetector'),
    ('backend.utils.audio_utils', 'AudioProcessor'),
]

for module_name, class_name in modules:
    try:
        module = __import__(module_name, fromlist=[class_name])
        getattr(module, class_name)
        print(f"   ✅ {class_name}")
    except Exception as e:
        print(f"   ❌ {class_name}: {e}")

# Test 3: Check models directory
print("\n3. Checking models directory...")
if os.path.exists('models'):
    print("   ✅ models/ directory exists")
    if os.path.exists('models/face_emotion_model.pth'):
        print("   ✅ face_emotion_model.pth exists")
    else:
        print("   ⚠️  face_emotion_model.pth not found (need to train)")
    if os.path.exists('models/audio_emotion_model.pth'):
        print("   ✅ audio_emotion_model.pth exists")
    else:
        print("   ⚠️  audio_emotion_model.pth not found (need to train)")
else:
    print("   ❌ models/ directory missing")

# Test 4: Check notebooks
print("\n4. Checking training notebooks...")
notebooks = [
    'notebooks/train_face_emotion.ipynb',
    'notebooks/train_audio_emotion.ipynb'
]
for nb in notebooks:
    if os.path.exists(nb):
        print(f"   ✅ {nb}")
    else:
        print(f"   ❌ {nb} missing")

# Test 5: Check Streamlit app
print("\n5. Checking Streamlit app...")
if os.path.exists('app/streamlit_app.py'):
    print("   ✅ streamlit_app.py exists")
else:
    print("   ❌ streamlit_app.py missing")

print("\n" + "=" * 50)
print("✅ Project structure test complete!")
print("\nNext steps:")
print("  1. Train models using Jupyter notebooks")
print("  2. Start FastAPI: uvicorn backend.api:app --reload")
print("  3. Start Streamlit: streamlit run app/streamlit_app.py")

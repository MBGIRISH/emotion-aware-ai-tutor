"""
Real-time face emotion inference using trained FER-2013 model.
Uses MediaPipe for face detection and landmark extraction.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import base64
from PIL import Image
import io
import os
from dotenv import load_dotenv

from backend.utils.face_detector import FaceDetector
from backend.utils.logger import setup_logger

load_dotenv()
logger = setup_logger(__name__)

# Emotion class labels (FER-2013)
EMOTION_LABELS = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral"
}


class EmotionCNN(nn.Module):
    """CNN architecture for face emotion recognition (FER-2013)"""
    
    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        # After 4 maxpools: 48->24->12->6->3, so 128 * 3 * 3 = 1152
        self.fc1 = nn.Linear(128 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # 48->24
        x = self.pool(torch.relu(self.conv2(x)))  # 24->12
        x = self.pool(torch.relu(self.conv3(x)))  # 12->6
        x = self.pool(torch.relu(self.conv4(x)))  # 6->3
        
        x = x.view(x.size(0), -1)  # Flatten dynamically
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class FaceEmotionInference:
    """Real-time face emotion inference engine"""
    
    def __init__(self, model_path: str = "models/face_emotion_model.pth", device: Optional[str] = None):
        """
        Initialize face emotion inference.
        
        Args:
            model_path: Path to trained PyTorch model
            device: 'cuda' or 'cpu' (auto-detected if None)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize face detector (MediaPipe)
        self.face_detector = FaceDetector()
        
        # Load emotion model
        self.model = EmotionCNN(num_classes=7)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            logger.info(f"Loaded face emotion model from {model_path}")
        else:
            logger.warning(f"Model not found at {model_path}. Using untrained model.")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing
        self.input_size = (48, 48)
    
    def preprocess_face(self, face_image: np.ndarray) -> torch.Tensor:
        """
        Preprocess face image for model input.
        
        Args:
            face_image: BGR image from OpenCV
            
        Returns:
            Preprocessed tensor (1, 1, 48, 48)
        """
        # Convert BGR to grayscale
        if len(face_image.shape) == 3:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_image
        
        # Resize to 48x48
        resized = cv2.resize(gray, self.input_size)
        
        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch + channel dimensions
        tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device)
    
    def predict(self, frame: np.ndarray) -> Dict[str, float]:
        """
        Predict emotions from a video frame.
        
        Args:
            frame: BGR image frame from webcam
            
        Returns:
            Dictionary of emotion probabilities
        """
        # Detect face
        faces = self.face_detector.detect_faces(frame)
        
        if not faces:
            # Return neutral if no face detected
            return {label: 0.0 if label != "neutral" else 1.0 for label in EMOTION_LABELS.values()}
        
        # Use first detected face
        face_bbox = faces[0]
        x, y, w, h = face_bbox
        
        # Extract face region
        face_roi = frame[y:y+h, x:x+w]
        
        # Preprocess
        face_tensor = self.preprocess_face(face_roi)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(face_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            probs = probabilities[0].cpu().numpy()
        
        # Convert to dictionary
        emotion_dict = {
            EMOTION_LABELS[i]: float(probs[i])
            for i in range(len(EMOTION_LABELS))
        }
        
        return emotion_dict
    
    def predict_from_base64(self, base64_string: str) -> Dict[str, float]:
        """
        Predict emotions from base64 encoded image.
        
        Args:
            base64_string: Base64 encoded image string
            
        Returns:
            Dictionary of emotion probabilities
        """
        try:
            # Decode base64
            image_data = base64.b64decode(base64_string)
            image = Image.open(io.BytesIO(image_data))
            
            # Convert PIL to OpenCV format
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            return self.predict(frame)
        except Exception as e:
            logger.error(f"Error processing base64 image: {e}")
            return {label: 0.0 if label != "neutral" else 1.0 for label in EMOTION_LABELS.values()}
    
    def run_realtime(self, camera_index: int = 0):
        """
        Run real-time emotion detection from webcam.
        
        Args:
            camera_index: Camera device index
        """
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            logger.error(f"Failed to open camera {camera_index}")
            return
        
        logger.info("Starting real-time face emotion detection. Press 'q' to quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Predict emotions
            emotions = self.predict(frame)
            
            # Draw face bounding box and emotions
            faces = self.face_detector.detect_faces(frame)
            if faces:
                x, y, w, h = faces[0]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Display top emotion
                top_emotion = max(emotions, key=emotions.get)
                cv2.putText(
                    frame,
                    f"{top_emotion}: {emotions[top_emotion]:.2f}",
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
            
            cv2.imshow("Face Emotion Detection", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Test real-time inference
    model_path = os.getenv("FACE_MODEL_PATH", "models/face_emotion_model.pth")
    inference = FaceEmotionInference(model_path=model_path)
    inference.run_realtime()


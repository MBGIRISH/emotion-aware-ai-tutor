#!/usr/bin/env python3
"""
Fixed training script for both face and audio emotion models.
"""

import sys
import os
sys.path.append('.')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from torchvision import transforms
from sklearn.metrics import classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.preprocessing_face import load_fer2013_data
from utils.preprocessing_audio import load_ravdess_data
from sklearn.model_selection import train_test_split

print("=" * 60)
print("TRAINING EMOTION-AWARE AI TUTOR MODELS")
print("=" * 60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\nUsing device: {device}\n')

# ============================================================================
# TRAIN FACE EMOTION MODEL
# ============================================================================
print("\n" + "=" * 60)
print("TRAINING FACE EMOTION MODEL (FER-2013)")
print("=" * 60)

print("\n1. Loading FER-2013 data...")
train_data, train_labels, test_data, test_labels = load_fer2013_data()
print(f"   Training: {len(train_data)} samples")
print(f"   Test: {len(test_data)} samples")

# Convert to uint8
train_data = (train_data * 255).astype('uint8')
test_data = (test_data * 255).astype('uint8')
train_labels = np.array(train_labels, dtype=np.int64)
test_labels = np.array(test_labels, dtype=np.int64)

# Dataset class
class FERDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = Image.fromarray(self.data[idx], mode='L')
        label = int(self.labels[idx])
        if self.transform:
            image = self.transform(image)
        return image, label

# Model
class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        # After 4 maxpools: 48 -> 24 -> 12 -> 6 -> 3, so 128 * 3 * 3 = 1152
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

# Create datasets
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

train_dataset = FERDataset(train_data, train_labels, transform=transform_train)
test_dataset = FERDataset(test_data, test_labels, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=False)

# Initialize model
print("\n2. Initializing model...")
model = EmotionCNN(num_classes=7).to(device)
print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Training
print("\n3. Training model...")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

num_epochs = 30
print(f"   Training for {num_epochs} epochs...")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        # Verify batch sizes match
        if images.size(0) != labels.size(0):
            print(f"Warning: Batch size mismatch at batch {batch_idx}: images={images.size(0)}, labels={labels.size(0)}")
            continue
        
        optimizer.zero_grad()
        outputs = model(images)
        
        # Debug: Check output and label shapes
        if batch_idx == 0 and epoch == 0:
            print(f"   Debug: images shape={images.shape}, outputs shape={outputs.shape}, labels shape={labels.shape}")
        
        if outputs.size(0) != labels.size(0):
            print(f"   Error: Output batch size {outputs.size(0)} != label batch size {labels.size(0)}")
            print(f"   Images shape: {images.shape}, Outputs shape: {outputs.shape}")
            continue
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    scheduler.step()
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"   Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

# Evaluation
print("\n4. Evaluating on test set...")
model.eval()
test_correct = 0
test_total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        if images.size(0) != labels.size(0):
            continue
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_accuracy = 100 * test_correct / test_total
print(f"   Test Accuracy: {test_accuracy:.2f}%")

# Save model
print("\n5. Saving model...")
model_path = 'models/face_emotion_model.pth'
os.makedirs(os.path.dirname(model_path), exist_ok=True)
torch.save(model.state_dict(), model_path)
print(f"   ✅ Saved to {model_path}")

# ============================================================================
# TRAIN AUDIO EMOTION MODEL
# ============================================================================
print("\n" + "=" * 60)
print("TRAINING AUDIO EMOTION MODEL (RAVDESS)")
print("=" * 60)

print("\n1. Loading RAVDESS data...")
X, y = load_ravdess_data()
print(f"   Total samples: {len(X)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Training: {len(X_train)} samples")
print(f"   Test: {len(X_test)} samples")

class AudioEmotionDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = np.array(labels, dtype=np.int64)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = torch.FloatTensor(self.features[idx])
        label = int(self.labels[idx])
        return feature, label

class AudioEmotionLSTM(nn.Module):
    def __init__(self, input_size=13, hidden_size=128, num_layers=2, num_classes=8):
        super(AudioEmotionLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=0.3, bidirectional=True
        )
        self.fc1 = nn.Linear(hidden_size * 2, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        x = torch.relu(self.fc1(last_output))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

train_dataset = AudioEmotionDataset(X_train, y_train)
test_dataset = AudioEmotionDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

print("\n2. Initializing model...")
input_size = X_train[0].shape[1] if len(X_train) > 0 else 13
model = AudioEmotionLSTM(input_size=input_size, num_classes=8).to(device)
print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")

print("\n3. Training model...")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

num_epochs = 50
print(f"   Training for {num_epochs} epochs...")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    scheduler.step()
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"   Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

print("\n4. Evaluating on test set...")
model.eval()
test_correct = 0
test_total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for features, labels in test_loader:
        features, labels = features.to(device), labels.to(device)
        outputs = model(features)
        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_accuracy = 100 * test_correct / test_total
print(f"   Test Accuracy: {test_accuracy:.2f}%")

print("\n5. Saving model...")
model_path = 'models/audio_emotion_model.pth'
os.makedirs(os.path.dirname(model_path), exist_ok=True)
torch.save(model.state_dict(), model_path)
print(f"   ✅ Saved to {model_path}")

print("\n" + "=" * 60)
print("✅ TRAINING COMPLETE!")
print("=" * 60)
print("\nModels saved to:")
print("  - models/face_emotion_model.pth")
print("  - models/audio_emotion_model.pth")
print("\nYou can now start the system!")


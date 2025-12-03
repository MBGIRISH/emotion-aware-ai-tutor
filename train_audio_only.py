#!/usr/bin/env python3
"""
Train only the audio emotion model (RAVDESS).
Use this if face model is already trained.
"""

import sys
import os
sys.path.append('.')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.preprocessing_audio import load_ravdess_data

print("=" * 60)
print("TRAINING AUDIO EMOTION MODEL (RAVDESS)")
print("=" * 60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\nUsing device: {device}\n')

# Load data
print("1. Loading RAVDESS data...")
X, y = load_ravdess_data()
print(f"   Total samples: {len(X)}")

# Split
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
        # Convert list to numpy array, then to tensor
        feature = np.array(self.features[idx])
        feature = torch.FloatTensor(feature)
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
# X_train is a list of lists, convert to numpy to get shape
if len(X_train) > 0:
    # Convert first sample to numpy array
    first_sample = np.array(X_train[0])
    if len(first_sample.shape) == 2:
        input_size = first_sample.shape[1]  # (sequence_length, features)
    elif len(first_sample.shape) == 1:
        input_size = first_sample.shape[0]
    else:
        input_size = 13
else:
    input_size = 13
print(f"   Input size: {input_size}")
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
print("✅ AUDIO MODEL TRAINING COMPLETE!")
print("=" * 60)
print("\nBoth models are now ready:")
print("  ✅ models/face_emotion_model.pth")
print("  ✅ models/audio_emotion_model.pth")
print("\nYou can now start the system!")


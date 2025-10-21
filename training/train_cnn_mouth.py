import sys
sys.path.append('../utils')
from dataset_loader import get_dataloaders

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import pandas as pd

# ==== CẤU HÌNH ====
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 0.0001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = "models/cnn_mouth.pth"
PLOT_PATH = "models/cnn_mouth_training_curve.png"

# ==== ĐÁNH GIÁ MÔ HÌNH ====
def evaluate_model(model, val_loader):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE).float().unsqueeze(1)

            outputs = model(inputs)
            preds = (outputs > 0.5).float()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Flatten lists
    all_labels = np.array(all_labels).flatten()
    all_preds = np.array(all_preds).flatten()

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("\n=== Confusion Matrix ===")
    print(cm)

    # Classification Report
    report = classification_report(all_labels, all_preds, target_names=["Class 0", "Class 1"], output_dict=True)
    print("\n=== Classification Report ===")
    print(classification_report(all_labels, all_preds, target_names=["Class 0", "Class 1"]))

    # Generate Summary Table
    summary_table = pd.DataFrame({
        "CNN Based on": ["ResNet18"],
        "Class Name": ["Binary (0/1)"],
        "Precision": [f"{(report['Class 1']['precision']):.4f}"],
        "Recall": [f"{(report['Class 1']['recall']):.4f}"],
        "F1-Score": [f"{(report['Class 1']['f1-score']):.4f}"],
        "Accuracy": [f"{(report['accuracy']):.4f}"]
    })

    print("\n=== Summary Table ===")
    print(summary_table.to_string(index=False))

    return cm, report, summary_table

# ==== HÀM HUẤN LUYỆN ====
def train(model, train_loader, val_loader):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.to(DEVICE)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        correct = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.float().to(DEVICE).unsqueeze(1)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()

        train_loss = total_loss / len(train_loader)
        train_acc = correct / len(train_loader.dataset)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # Validation
        model.eval()
        total_val_loss = 0
        val_correct = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(DEVICE)
                labels = labels.float().to(DEVICE).unsqueeze(1)

                outputs = model(inputs)
                val_loss = criterion(outputs, labels)
                total_val_loss += val_loss.item()

                preds = (outputs > 0.5).float()
                val_correct += (preds == labels).sum().item()

        val_loss = total_val_loss / len(val_loader)
        val_acc = val_correct / len(val_loader.dataset)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} - Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} - Acc: {val_acc:.4f}")

    # === VẼ BIỂU ĐỒ ===
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss over Epochs")

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Acc')
    plt.plot(val_accuracies, label='Val Acc')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy over Epochs")

    plt.tight_layout()
    plt.grid(True)
    os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)
    plt.savefig(PLOT_PATH)
    plt.show()

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"📦 Model saved to {MODEL_SAVE_PATH}")
    print(f"📊 Training curve saved to {PLOT_PATH}")

# ==== MAIN ====
if __name__ == "__main__":
    print("🔄 Loading mouth dataset...")
    train_loader, val_loader = get_dataloaders(
        "C:/code/Project3month/driver_fatigue_detection_new/data/processed/mouth",
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        grayscale=False
    )

    print("🧠 Loading pretrained ResNet18...")
    base_model = models.resnet18(pretrained=True)
    base_model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    base_model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(base_model.fc.in_features, 1),
        nn.Sigmoid()
    )

    print("🚀 Training CNN on Yawn Detection...")
    train(base_model, train_loader, val_loader)

    # ==== ĐÁNH GIÁ CUỐI ====
    print("🔍 Evaluating model...")
    evaluate_model(base_model, val_loader)

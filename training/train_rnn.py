import os
import csv
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split

# ==== CẤU HÌNH ====
SEQUENCE_LENGTH = 30
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
LOG_PATH = "C:/code/Project3month/driver_fatigue_detection_new/data/logs"
MODEL_SAVE_PATH = "models/rnn_drowsy.pth"
PLOT_PATH = "models/rnn_training_curve.png"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== DATASET ====
def encode_state(eye, mouth):
    eye_enc = 1 if eye == "Closed" else 0
    mouth_enc = 1 if mouth == "Yawning" else 0
    return [eye_enc, mouth_enc]

class LogSequenceDataset(Dataset):
    def __init__(self, log_dir, sequence_length=30):
        self.samples = []
        logs = glob(os.path.join(log_dir, "*.csv"))

        for log_file in logs:
            with open(log_file, 'r') as f:
                reader = csv.DictReader(f)
                sequence = []
                for row in reader:
                    encoded = encode_state(row["eye_state"], row["mouth_state"])
                    sequence.append(encoded)

            for i in range(0, len(sequence) - sequence_length):
                seq = sequence[i:i+sequence_length]
                drowsy_count = sum([1 for x in seq if x[0] == 1 or x[1] == 1])
                label = 1 if drowsy_count >= 5 else 0
                self.samples.append((torch.tensor(seq, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# ==== MODEL ====
class DrowsinessRNN(nn.Module):
    def __init__(self, input_size=2, hidden_size=64):
        super(DrowsinessRNN, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return self.sigmoid(out)

# ==== TRAINING ====
def train():
    dataset = LogSequenceDataset(LOG_PATH, sequence_length=SEQUENCE_LENGTH)
    print(f"\n🔍 Total samples: {len(dataset)}")

    train_indices, val_indices = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=42)
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = DrowsinessRNN().to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE).unsqueeze(1)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / len(train_loader)
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        model.eval()
        val_loss_total, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE).unsqueeze(1)

                outputs = model(inputs)
                val_loss = criterion(outputs, labels)
                val_loss_total += val_loss.item()

                preds = (outputs > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss_avg = val_loss_total / len(val_loader)
        val_acc = val_correct / val_total
        val_losses.append(val_loss_avg)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} - Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss_avg:.4f} - Acc: {val_acc:.4f}")

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
    os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)
    plt.savefig(PLOT_PATH)
    plt.show()

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\n✅ RNN model saved to {MODEL_SAVE_PATH}")
    print(f"📊 Training curve saved to {PLOT_PATH}")

if __name__ == "__main__":
    train()

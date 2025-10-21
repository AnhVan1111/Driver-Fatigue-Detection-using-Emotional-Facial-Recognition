import torch
import torch.nn as nn
import os

# ==== CẤU HÌNH ====
MODEL_PATH = "C:/code/Project3month/driver_fatigue_detection_new/training/models/rnn_drowsy.pth"
SEQUENCE_LENGTH = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== RNN MODEL (PHẢI giống với train_rnn.py) ====
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

# ==== LOAD MODEL ====
model = DrowsinessRNN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ==== DỰ ĐOÁN ====
def predict_drowsiness(sequence):
    """
    sequence: List of [eye_state, mouth_state], length = 30
              eye: 0=open, 1=closed
              mouth: 0=not yawning, 1=yawning
    Return: True if drowsy, else False
    """
    if len(sequence) != SEQUENCE_LENGTH:
        raise ValueError(f"Input sequence must be {SEQUENCE_LENGTH} in length.")

    input_tensor = torch.tensor([sequence], dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        output = model(input_tensor)
        prediction = (output.item() > 0.5)
    return prediction

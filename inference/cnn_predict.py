import torch
import torch.nn as nn
from torchvision import models

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EYE_MODEL_PATH = "C:/code/Project3month/driver_fatigue_detection_new/training/models/cnn_eye.pth"
MOUTH_MODEL_PATH = "C:/code/Project3month/driver_fatigue_detection_new/training/models/cnn_mouth.pth"

def create_model():
    model = models.resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Sequential(
        nn.Dropout(0.3),              # Phải có đúng nếu đã train với dropout
        nn.Linear(model.fc.in_features, 1),
        nn.Sigmoid()
    )
    return model

def load_cnn_model(model_path):
    model = create_model()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

eye_model = load_cnn_model(EYE_MODEL_PATH)
mouth_model = load_cnn_model(MOUTH_MODEL_PATH)

# ==== PREDICT ====

def predict_eye_state(left_eye_tensor, right_eye_tensor):
    """
    Dự đoán trạng thái mắt từ 2 tensor mắt trái và phải
    Return: 'Open' or 'Closed'
    """
    inputs = torch.stack([left_eye_tensor, right_eye_tensor]).to(DEVICE)
    with torch.no_grad():
        outputs = eye_model(inputs)
        preds = (outputs > 0.5).float().cpu().numpy()

    # Nếu cả 2 mắt đều nhắm thì coi là Closed
    if preds.sum() == 0:
        return "Closed"
    return "Open"
def predict_mouth_state(mouth_tensor):
    """
    Dự đoán trạng thái miệng từ tensor miệng
    Return: 'Yawning' or 'Not Yawning'
    """
    input_tensor = mouth_tensor.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = mouth_model(input_tensor)
        pred = (output > 0.5).float().item()
    return "Yawning" if pred == 1.0 else "Not Yawning"

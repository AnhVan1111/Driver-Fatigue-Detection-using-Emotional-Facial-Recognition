import sys
sys.path.append('../utils')
from crop_utils import extract_eye_mouth_regions

import cv2
import torch
import numpy as np
import mediapipe as mp
from datetime import datetime
import csv
import os

from inference.cnn_predict import predict_eye_state, predict_mouth_state
from inference.rnn_predict import predict_drowsiness


# ==== CẤU HÌNH ====
SEQUENCE_LENGTH = 30
SAVE_LOG = True
LOG_DIR = "C:/code/Project3month/driver_fatigue_detection_new/data/logs"
os.makedirs(LOG_DIR, exist_ok=True)

# ==== KHỞI TẠO ====
cap = cv2.VideoCapture(0)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Tạo log file nếu cần
if SAVE_LOG:
    log_file = os.path.join(LOG_DIR, f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    f = open(log_file, mode='w', newline='')
    writer = csv.writer(f)
    writer.writerow(['timestamp', 'eye_state', 'mouth_state'])

def generate_video_stream():
    cap = cv2.VideoCapture(0)
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

    buffer = []
    SEQUENCE_LENGTH = 30

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Xử lý ảnh và dự đoán
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        if result.multi_face_landmarks:
            landmarks = result.multi_face_landmarks[0]

            left_eye, right_eye, mouth_img = extract_eye_mouth_regions(frame)
            if left_eye is None or right_eye is None or mouth_img is None:
                continue

            eye_state = predict_eye_state(left_eye, right_eye).capitalize()
            if eye_state not in ['Open', 'Closed']:
                continue

            mouth_state = predict_mouth_state(mouth_img)

            # Hiển thị trạng thái
            cv2.putText(frame, f"Eye: {eye_state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(frame, f"Mouth: {mouth_state}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            eye_enc = 1 if eye_state == "Closed" else 0
            mouth_enc = 1 if mouth_state == "Yawning" else 0
            buffer.append([eye_enc, mouth_enc])
            if len(buffer) > SEQUENCE_LENGTH:
                buffer.pop(0)

            if len(buffer) == SEQUENCE_LENGTH:
                drowsy = predict_drowsiness(buffer)
                if drowsy:
                    cv2.putText(frame, "⚠️ DROWSINESS DETECTED!", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

        # Encode JPEG và gửi về web
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

cap.release()
cv2.destroyAllWindows()
if SAVE_LOG:
    f.close()



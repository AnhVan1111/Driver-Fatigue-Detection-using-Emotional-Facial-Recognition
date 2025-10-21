import cv2
import numpy as np
import mediapipe as mp
import torch
from torchvision import transforms

mp_face_mesh = mp.solutions.face_mesh
FACE_MESH = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Landmark indices
LEFT_EYE_IDX = [33, 133, 160, 159, 158, 157, 173, 144, 145, 153, 154, 155]
RIGHT_EYE_IDX = [362, 263, 387, 386, 385, 384, 398, 373, 374, 380, 381, 382]
MOUTH_IDX = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308,
    324, 318, 402, 317, 14, 87, 178, 88, 95, 185, 40, 39,
    37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82,
    81, 42, 183, 78]

# Preprocessing transform (resize + normalize)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def get_landmark_box(landmarks, indices, frame_width, frame_height, padding=10):
    """
    Tính bounding box từ danh sách index landmark.
    """
    x_coords = [int(landmarks[i].x * frame_width) for i in indices]
    y_coords = [int(landmarks[i].y * frame_height) for i in indices]

    x_min = max(min(x_coords) - padding, 0)
    x_max = min(max(x_coords) + padding, frame_width)
    y_min = max(min(y_coords) - padding, 0)
    y_max = min(max(y_coords) + padding, frame_height)

    return x_min, y_min, x_max, y_max

def extract_eye_mouth_regions(frame):
    """
    Từ 1 frame video: trích landmarks, crop 3 vùng: mắt trái, mắt phải, miệng
    Trả về tensor đã xử lý sẵn (sẵn sàng đưa vào CNN)
    """
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = FACE_MESH.process(rgb_frame)

    if not results.multi_face_landmarks:
        return None, None, None

    landmarks = results.multi_face_landmarks[0].landmark

    # Crop left eye
    left_box = get_landmark_box(landmarks, LEFT_EYE_IDX, w, h)
    left_eye = frame[left_box[1]:left_box[3], left_box[0]:left_box[2]]
    left_eye_tensor = transform(left_eye)

    # Crop right eye
    right_box = get_landmark_box(landmarks, RIGHT_EYE_IDX, w, h)
    right_eye = frame[right_box[1]:right_box[3], right_box[0]:right_box[2]]
    right_eye_tensor = transform(right_eye)

    # Crop mouth
    mouth_box = get_landmark_box(landmarks, MOUTH_IDX, w, h)
    mouth = frame[mouth_box[1]:mouth_box[3], mouth_box[0]:mouth_box[2]]
    mouth_tensor = transform(mouth)

    return left_eye_tensor, right_eye_tensor, mouth_tensor

# def show_landmarks_from_camera():
#     cap = cv2.VideoCapture(0)

#     if not cap.isOpened():
#         print("Không mở được camera.")
#         return

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         h, w, _ = frame.shape
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = FACE_MESH.process(rgb_frame)

#         if results.multi_face_landmarks:
#             landmarks = results.multi_face_landmarks[0].landmark

#             # Vẽ điểm mắt trái
#             for idx in LEFT_EYE_IDX:
#                 x = int(landmarks[idx].x * w)
#                 y = int(landmarks[idx].y * h)
#                 cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

#             # Vẽ điểm mắt phải
#             for idx in RIGHT_EYE_IDX:
#                 x = int(landmarks[idx].x * w)
#                 y = int(landmarks[idx].y * h)
#                 cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

#             # Vẽ điểm miệng
#             for idx in MOUTH_IDX:
#                 x = int(landmarks[idx].x * w)
#                 y = int(landmarks[idx].y * h)
#                 cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

#         # Hiển thị ảnh có overlay điểm landmark
#         cv2.imshow("Face Landmarks", frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# # Gọi hàm để chạy
# show_landmarks_from_camera()



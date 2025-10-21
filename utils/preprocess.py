import os
import cv2
import shutil
import random

# Kích thước ảnh chuẩn
IMG_SIZE = (64, 64)
# Tỷ lệ train/val
SPLIT_RATIO = 0.8

def prepare_and_split_dataset(src_folder, dest_folder):
    """
    Hàm xử lý ảnh từ thư mục src_folder (theo nhãn con)
    và lưu về dest_folder theo cấu trúc train/val/[label]/
    """
    for label_name in os.listdir(src_folder):
        label_path = os.path.join(src_folder, label_name)
        if not os.path.isdir(label_path):
            continue

        # Lấy danh sách ảnh, shuffle
        images = os.listdir(label_path)
        random.shuffle(images)
        split_idx = int(len(images) * SPLIT_RATIO)
        train_imgs = images[:split_idx]
        test_imgs = images[split_idx:]

        # Lưu ảnh train/test vào thư mục mới
        for subset, subset_imgs in zip(['train', 'val'], [train_imgs, test_imgs]):
            save_dir = os.path.join(dest_folder, subset, label_name)
            os.makedirs(save_dir, exist_ok=True)

            for img_name in subset_imgs:
                src_img_path = os.path.join(label_path, img_name)
                dst_img_path = os.path.join(save_dir, img_name)

                # Đọc và resize ảnh
                img = cv2.imread(src_img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, IMG_SIZE)
                    cv2.imwrite(dst_img_path, img)

if __name__ == "__main__":
    print("🔄 Splitting & preprocessing eye dataset...")
    prepare_and_split_dataset("C:/code/Project3month/driver_fatigue_detection_new/data/eye", "C:/code/Project3month/driver_fatigue_detection_new/data/processed/eye")

    print("🔄 Splitting & preprocessing mouth dataset...")
    prepare_and_split_dataset("C:/code/Project3month/driver_fatigue_detection_new/data/mouth", "C:/code/Project3month/driver_fatigue_detection_new/data/processed/mouth")

    print("✅ Done. Processed data saved in: data/processed/")

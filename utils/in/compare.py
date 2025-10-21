import os
import random
import cv2
import matplotlib.pyplot as plt

# === Base directory paths ===
base_original_dir = 'C:/code/Project3month/driver_fatigue_detection_new/data'
base_processed_dir = 'C:/code/Project3month/driver_fatigue_detection_new/data/processed'

# === Categories to visualize ===
categories = {
    'eye': ['open', 'closed'],
    'mouth': ['yawn', 'no_yawn']
}

for region, statuses in categories.items():
    for status in statuses:
        # Original image path
        original_dir = os.path.join(base_original_dir, region, status)

        # Processed image path (choose 'train' for display)
        processed_dir = os.path.join(base_processed_dir, region, 'train', status)

        # Check if both directories exist
        if not os.path.isdir(original_dir) or not os.path.isdir(processed_dir):
            print(f"❌ Missing directory: {original_dir} or {processed_dir}")
            continue

        # Get common files between original and processed
        original_files = set(os.listdir(original_dir))
        processed_files = set(os.listdir(processed_dir))
        common_files = list(original_files & processed_files)

        if len(common_files) == 0:
            print(f"⚠️ No matching images found in: {original_dir} and {processed_dir}")
            continue

        # Pick up to 5 samples
        sample_files = random.sample(common_files, min(5, len(common_files)))

        # Plot 5 original + processed image pairs (2 rows, 5 columns)
        plt.figure(figsize=(15, 4))
        for i, filename in enumerate(sample_files):
            # Original image
            orig_path = os.path.join(original_dir, filename)
            orig_img = cv2.imread(orig_path)
            if orig_img is None:
                print(f"⚠️ Could not read original image: {orig_path}")
                continue
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

            # Processed image (assumed grayscale)
            proc_path = os.path.join(processed_dir, filename)
            proc_img = cv2.imread(proc_path, cv2.IMREAD_GRAYSCALE)
            if proc_img is None:
                print(f"⚠️ Could not read processed image: {proc_path}")
                continue

            # Show original
            plt.subplot(2, 5, i + 1)
            plt.imshow(orig_img)
            plt.title(f"Original {i+1}", fontsize=8)
            plt.axis('off')

            # Show processed
            plt.subplot(2, 5, i + 6)
            plt.imshow(proc_img, cmap='gray')
            plt.title(f"Processed {i+1}", fontsize=8)
            plt.axis('off')

        plt.suptitle(f"Comparison: {region.upper()} - {status.upper()} (5 Samples)", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.92])
        plt.show()

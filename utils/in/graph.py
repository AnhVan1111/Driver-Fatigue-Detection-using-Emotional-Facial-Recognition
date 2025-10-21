import os
import matplotlib.pyplot as plt

# Path to processed image directories (eye or mouth)
base_dir = 'C:/code/Project3month/driver_fatigue_detection_new/data/processed/mouth'  # or '.../processed/mouth'

def count_images_by_label(base_dir):
    counts = {}
    for split in ['train', 'val']:
        split_path = os.path.join(base_dir, split)
        if not os.path.isdir(split_path):
            continue
        for label in os.listdir(split_path):
            label_path = os.path.join(split_path, label)
            if os.path.isdir(label_path):
                n_images = len(os.listdir(label_path))
                key = f"{split}/{label}"
                counts[key] = n_images
    return counts

# Count images
counts = count_images_by_label(base_dir)

# Plot the bar chart
labels = list(counts.keys())
values = list(counts.values())

plt.figure(figsize=(10, 5))
plt.bar(labels, values, color='skyblue')
plt.xticks(rotation=45)
plt.ylabel('Number of Images')
plt.title('Distribution of Mouth Images After Processing')
plt.tight_layout()
plt.show()

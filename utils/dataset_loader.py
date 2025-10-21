import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(data_dir, img_size=(224, 224), batch_size=32, grayscale=False):
    """
    Load DataLoader từ thư mục data_dir (phải có train/ và val/)
    - img_size: kích thước ảnh resize
    - grayscale: True nếu ảnh là đen trắng (1 kênh), False nếu RGB (3 kênh)
    """

    # Chọn số kênh và mean/std theo chế độ màu
    if grayscale:
        num_channels = 1
        mean = [0.5]
        std = [0.5]
    else:
        num_channels = 3
        mean = [0.5] * 3
        std = [0.5] * 3

    # Transform cho ảnh train (có augmentation)
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.Resize(img_size),
        transforms.Grayscale(num_output_channels=1) if grayscale else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # Transform cho ảnh validation (không augment)
    val_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.Grayscale(num_output_channels=1) if grayscale else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # Load dataset từ thư mục
    train_path = os.path.join(data_dir, "train")
    val_path = os.path.join(data_dir, "val")

    train_dataset = datasets.ImageFolder(train_path, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_path, transform=val_transform)

    # Tạo DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

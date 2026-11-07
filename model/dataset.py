import torch, os
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225]
    )
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225]
    )
])

def load_data(data_dir="./data", batch_size=32, train_ratio=0.8):
    train_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(root=data_dir, transform=test_transform)

    class_names = train_dataset.classes

    train_size = int(train_ratio * len(train_dataset))
    test_size = len(train_dataset) - train_size

    train_set, _ = random_split(
        train_dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    _, test_set = random_split(
        test_dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader, class_names
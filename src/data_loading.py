import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


class FashionMNISTData:
    def __init__(self, batch_size=128, val_split=0.1, root="./data", seed=42):
        self.batch_size = batch_size
        self.val_split = val_split
        self.root = root
        self.seed = seed

        # Step 1 and 2: Load and normalize
        transform = transforms.ToTensor()
        full_train = datasets.FashionMNIST(root=self.root, train=True, transform=transform, download=True)
        test = datasets.FashionMNIST(root=self.root, train=False, transform=transform, download=True)

        # Step 3: Split train/val
        total_train = len(full_train)
        val_size = int(total_train * val_split)
        train_size = total_train - val_size
        torch.manual_seed(seed)
        train, val = random_split(full_train, [train_size, val_size])

        # Dataloaders
        self.train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import rasterio
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from scipy.ndimage import gaussian_filter, laplace, uniform_filter

# === 1. Data Paths and Splitting ===
positive_dir = "./01_samples/tif/dem/positive"
negative_dir = "./01_samples/tif/dem/negative"
pos_files = [os.path.join(positive_dir, f) for f in os.listdir(positive_dir) if f.endswith('.tif')]
neg_files = [os.path.join(negative_dir, f) for f in os.listdir(negative_dir) if f.endswith('.tif')]
samples = [(f, 1) for f in pos_files] + [(f, 0) for f in neg_files]

# 15% for test set; then 15% of remaining for validation (i.e., total 15%)
train_val, test_samples = train_test_split(samples, test_size=0.15, stratify=[lbl for _, lbl in samples], random_state=42)
train_samples, val_samples = train_test_split(train_val, test_size=0.1765, stratify=[lbl for _, lbl in train_val], random_state=42)

# === 2. Multi-channel DEM Derived Feature Dataset ===
class DEMMultiChannelDataset(Dataset):
    def __init__(self, samples, augment=False, sigma=1.0):
        self.samples = samples
        self.augment = augment
        self.sigma = sigma

        self.aug = transforms.Compose([
            transforms.RandomCrop(56),
            transforms.Resize(64),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float32)
        arr = np.nan_to_num(arr)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)

        # Generate derived features: smooth, slope, Laplacian, local std
        smooth = gaussian_filter(arr, sigma=self.sigma)
        grad_y, grad_x = np.gradient(smooth)
        slope = np.sqrt(grad_x ** 2 + grad_y ** 2)
        lap = laplace(smooth)

        mean = uniform_filter(smooth, size=5)
        mean_sq = uniform_filter(smooth ** 2, size=5)
        std = np.sqrt(np.clip(mean_sq - mean ** 2, a_min=0.0, a_max=None))

        stack = np.stack([arr, smooth, slope, lap, std], axis=0)

        # Normalize each channel
        for i in range(stack.shape[0]):
            mean = stack[i].mean()
            std = stack[i].std()
            stack[i] = (stack[i] - mean) / (std + 1e-6)

        img = torch.tensor(stack, dtype=torch.float32)
        if self.augment:
            img = self.aug(img)
        return img, torch.tensor(label, dtype=torch.long)

# === 3. ResNet18 with 5-Channel Input ===
class ResNet5Band(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet18(pretrained=False)
        base.conv1 = nn.Conv2d(5, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base.maxpool = nn.Identity()  # Remove maxpool to preserve spatial resolution
        base.fc = nn.Sequential(
            nn.BatchNorm1d(base.fc.in_features),
            nn.Dropout(0.4),
            nn.Linear(base.fc.in_features, 2)
        )
        self.model = base

    def forward(self, x):
        return self.model(x)

# === 4. Single Epoch Training ===
def run_epoch(model, loader, optimizer, criterion, train=True):
    model.train() if train else model.eval()
    correct, total, running_loss = 0, 0, 0.0
    loop = tqdm(loader, desc="Train" if train else "Val", leave=False)
    for x, y in loop:
        x, y = x.to(device), y.to(device)
        if train: optimizer.zero_grad()
        with torch.set_grad_enabled(train):
            out = model(x)
            loss = criterion(out, y)
            if train:
                loss.backward()
                optimizer.step()
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        running_loss += loss.item() * x.size(0)
        loop.set_postfix(loss=loss.item(), acc=correct/total)
    return running_loss / total, correct / total

# === 5. Model Initialization ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNet5Band().to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

train_loader = DataLoader(DEMMultiChannelDataset(train_samples, augment=True), batch_size=32, shuffle=True)
val_loader   = DataLoader(DEMMultiChannelDataset(val_samples), batch_size=32)
test_loader  = DataLoader(DEMMultiChannelDataset(test_samples), batch_size=32)

# === 6. Main Training Loop ===
best_val_acc = 0.0
for epoch in range(1, 51):
    print(f"\n📘 Epoch {epoch}")
    train_loss, train_acc = run_epoch(model, train_loader, optimizer, criterion, train=True)
    val_loss, val_acc = run_epoch(model, val_loader, optimizer, criterion, train=False)
    scheduler.step()
    print(f"📊 Train Loss: {train_loss:.4f}, Acc: {train_acc:.2%} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.2%}")
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model_7.pth")
        print("✅ Best model saved!")

# === 7. Final Test Evaluation ===
model.load_state_dict(torch.load("best_model_7.pth"))
model.eval()
_, test_acc = run_epoch(model, test_loader, optimizer=None, criterion=criterion, train=False)
print(f"\n🎯 Test Accuracy: {test_acc:.2%}")

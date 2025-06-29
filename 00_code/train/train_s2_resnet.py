import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import models
import torchvision.transforms.functional as F
from pathlib import Path
from dataset import SentinelDataset  # You should place this in 00_code/utils/dataset.py

# Optional visualization tool (not used in training loop below)
import matplotlib.pyplot as plt

# Minimum shape for cropping Sentinel-2 patches (H, W)
MIN_SHAPE = (206, 204)

# === 1. Data paths ===
# Use relative paths for project portability
pos_path = Path('./01_samples/tif/s2/positive')
neg_path = Path('./01_samples/tif/s2/negative')
positive_masked = Path('./01_samples/tif/s2_masked/positive')
negative_masked = Path('./01_samples/tif/s2_masked/negative')

# Get all image file paths
pos_files = [f for f in pos_path.glob("**/*") if f.is_file()]
neg_files = [f for f in neg_path.glob("**/*") if f.is_file()]

# Initialize dataset
dataset = SentinelDataset(
    pos_files=pos_files,
    neg_files=neg_files,
    pos_masked=positive_masked,
    neg_masked=negative_masked,
    transform=lambda x: F.center_crop(x, output_size=MIN_SHAPE)
)

# === 2. Train/Val split ===
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])

# DataLoaders
train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_set,   batch_size=64, shuffle=False, num_workers=0)

# === 3. Model Definition ===
def build_resnet(num_input_channels=5, num_classes=2):
    model = models.resnet18(weights=None)  # No pretrained weights
    orig_conv = model.conv1
    model.conv1 = nn.Conv2d(
        num_input_channels,
        orig_conv.out_channels,
        kernel_size=orig_conv.kernel_size,
        stride=orig_conv.stride,
        padding=orig_conv.padding,
        bias=orig_conv.bias is not None
    )
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = build_resnet(num_input_channels=5, num_classes=2).to(device)

# === 4. Loss & Optimizer ===
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

# === 5. Training loop ===
def train_epoch(loader):
    model.train()
    total_loss = total_correct = 0
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        preds = model(imgs)
        loss = criterion(preds, lbls)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        total_correct += (preds.argmax(1) == lbls).sum().item()
    return total_loss / len(loader.dataset), total_correct / len(loader.dataset)

# === 6. Evaluation loop ===
def eval_epoch(loader):
    model.eval()
    total_loss = total_correct = 0
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            preds = model(imgs)
            loss = criterion(preds, lbls)
            total_loss += loss.item() * imgs.size(0)
            total_correct += (preds.argmax(1) == lbls).sum().item()
    return total_loss / len(loader.dataset), total_correct / len(loader.dataset)

# === 7. Main training routine (uncomment to use) ===
# best_val_loss = float('inf')
# for epoch in range(1, 26):
#     tl, ta = train_epoch(train_loader)
#     vl, va = eval_epoch(val_loader)
#     scheduler.step(vl)
#     print(f'Epoch {epoch:02d} | '
#           f'Train loss {tl:.4f}, acc {ta:.4f} | '
#           f'Val loss {vl:.4f}, acc {va:.4f}')
#     if vl < best_val_loss:
#         best_val_loss = vl
#         torch.save(model.state_dict(), 'best_s2_resnet.pth')
#         print("✅ Model saved!")


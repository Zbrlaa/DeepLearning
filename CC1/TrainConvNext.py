import os
import torch
import torch.optim as optim
import torch.nn as nn
import copy
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import convnext_small, ConvNeXt_Small_Weights
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import leaderboard_client as lb

# ---------------- CONFIG ----------------
ROOT = "/scratch/Shawn/download/FCUNIST"
TRAIN_DIR = os.path.join(ROOT, "train")
os.environ['LEADERBOARD_TOKEN'] = 'e99def1e-e708-4c32-8591-78d6d27cd317'
os.environ['TORCH_HOME'] = '/scratch/Shawn/.cache/torch'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ---------------- WEIGHTS ----------------
weights = ConvNeXt_Small_Weights.IMAGENET1K_V1

# ---------------- TRANSFORMATIONS ----------------
# Base transforms recommandées par le modèle
base_tf = weights.transforms()

# Training avec augmentations supplémentaires
train_tf = transforms.Compose([
    transforms.Resize(232),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.1),
    transforms.Normalize(mean=base_tf.mean, std=base_tf.std)
])

val_tf = transforms.Compose([
    transforms.Resize(232),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=base_tf.mean, std=base_tf.std)
])

# ---------------- DATASETS ----------------
raw_dataset = datasets.ImageFolder(TRAIN_DIR)
targets = raw_dataset.targets
num_classes = len(os.listdir(TRAIN_DIR))
print(f"{num_classes} classes")

train_idx, val_idx = train_test_split(
    list(range(len(targets))),
    test_size=0.1,
    stratify=targets,
    random_state=42
)

train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
val_dataset = datasets.ImageFolder(TRAIN_DIR, transform=val_tf)

train_set = Subset(train_dataset, train_idx)
val_set = Subset(val_dataset, val_idx)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=4)

# ---------------- MODEL ----------------
model = convnext_small(weights=weights)
model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
model = model.to(device)

# ---------------- OPTIMIZER & LOSS ----------------
optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-2)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# ---------------- SCHEDULERS ----------------
num_epochs = 50
warmup_epochs = 5
warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs)
scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])

# ---------------- AMP ----------------
scaler = torch.cuda.amp.GradScaler()

# ---------------- EARLY STOPPING ----------------
best_val_loss = float('inf')
patience = 7
counter = 0
best_model_wts = copy.deepcopy(model.state_dict())

# ---------------- TRAINING LOOP ----------------
for epoch in range(num_epochs):
    model.train()
    total_loss, total, correct = 0, 0, 0
    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch} Train"):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits = model(imgs)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    train_acc = correct / total
    train_loss = total_loss / total

    # ---------------- VALIDATION ----------------
    model.eval()
    val_loss_sum, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch} Eval"):
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.cuda.amp.autocast():
                logits = model(imgs)
                loss = criterion(logits, labels)
            val_loss_sum += loss.item() * imgs.size(0)
            val_correct += (logits.argmax(1) == labels).sum().item()
            val_total += imgs.size(0)
    val_acc = val_correct / val_total
    val_loss = val_loss_sum / val_total
    print(f"Epoch {epoch}: Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Val Loss: {val_loss:.4f}")

    scheduler.step()

    # ---------------- EARLY STOPPING ----------------
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        counter = 0
        torch.save(best_model_wts, "/scratch/Shawn/best_model_convnext_small.pth")
    else:
        counter += 1
        print(f"EarlyStopping counter: {counter}/{patience}")
        if counter >= patience:
            print("Early stopping triggered!")
            break

    # ---------------- LEADERBOARD ----------------
    current_lr = float(optimizer.param_groups[0]['lr'])
    lb.submit_training(
        train_acc, train_loss,
        {"num_epochs": num_epochs, "lr": current_lr, "batch_size": 32, "epoch": epoch},
        tag=f"ConvNeXtSmall : epoch:{epoch}, lr:{current_lr:.6f}, batch_size:32"
    )
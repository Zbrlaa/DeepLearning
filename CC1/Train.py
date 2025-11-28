import os
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights
from tqdm import tqdm

#Config
ROOT = "/scratch/Shawn/download/FCUNIST"
TRAIN_DIR = os.path.join(ROOT, "train")
TEST_DIR = os.path.join(ROOT, "test")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


#Transformations
train_tf = transforms.Compose([
	transforms.Resize((224, 224)),
	transforms.RandomHorizontalFlip(),
	transforms.RandomRotation(10),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406],
						std=[0.229, 0.224, 0.225]),
])

val_tf = transforms.Compose([
	transforms.Resize((224, 224)),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406],
						std=[0.229, 0.224, 0.225]),
])


#Datasets
full_train = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)

train_size = int(0.9 * len(full_train))
val_size = len(full_train) - train_size
train_set, val_set = torch.utils.data.random_split(full_train, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=4)

num_classes = len(os.listdir(TRAIN_DIR))
print(f"{num_classes} classes")


#Model
weights = EfficientNet_V2_M_Weights.DEFAULT
model = efficientnet_v2_m(weights=weights)

model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

model = model.to(device)


#optim
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
criterion = nn.CrossEntropyLoss()

num_epochs = 10

train_losses = []
val_losses = []
train_accs = []
val_accs = []


#train
for epoch in range(num_epochs):
	model.train()
	total, correct, total_loss = 0, 0, 0

	for imgs, labels in tqdm(train_loader, desc=f"[Epoch {epoch}] Entrainement"):
		imgs, labels = imgs.to(device), labels.to(device)

		optimizer.zero_grad()
		logits = model(imgs)

		loss = criterion(logits, labels)
		loss.backward()
		optimizer.step()

		total_loss += loss.item() * imgs.size(0)
		preds = logits.argmax(dim=1)
		correct += (preds == labels).sum().item()
		total += imgs.size(0)

	train_acc = correct / total
	train_loss = total_loss / total

	#eval
	model.eval()
	val_correct, val_total, val_loss_sum = 0, 0, 0

	with torch.no_grad():
		for imgs, labels in tqdm(val_loader, desc=f"[Epoch {epoch}] Validation"):
			imgs, labels = imgs.to(device), labels.to(device)

			logits = model(imgs)
			loss = criterion(logits, labels)

			val_loss_sum += loss.item() * imgs.size(0)
			preds = logits.argmax(dim=1)
			val_correct += (preds == labels).sum().item()
			val_total += imgs.size(0)

	val_acc = val_correct / val_total
	val_loss = val_loss_sum / val_total

	print(f"Epoch {epoch}:")
	print(f"  Train     - acc={train_acc:.4f} loss={train_loss:.4f}")
	print(f"  Val       - acc={val_acc:.4f}   loss={val_loss:.4f}")
	print("-----------------------------------------------------")

	train_losses.append(train_loss)
	val_losses.append(val_loss)
	train_accs.append(train_acc)
	val_accs.append(val_acc)


plt.figure(figsize=(10,5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.title("Loss par epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10,5))
plt.plot(train_accs, label="Train Accuracy")
plt.plot(val_accs, label="Validation Accuracy")
plt.title("Accuracy par epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.show()
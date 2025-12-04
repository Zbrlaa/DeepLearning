import os
import torch
import torch.optim as optim
import torch.nn as nn
import copy
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
# from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights
from torchvision.models import efficientnet_v2_l, EfficientNet_V2_L_Weights
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import leaderboard_client as lb  # <- ajout pour lb.submit_training

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ['LEADERBOARD_TOKEN'] = 'e99def1e-e708-4c32-8591-78d6d27cd317'
os.environ['TORCH_HOME'] = '/scratch/Shawn/.cache/torch'

# --- DATASET ---
TRAIN_DIR = "/scratch/Shawn/download/FCUNIST/train"
num_classes = len(os.listdir(TRAIN_DIR))

raw_dataset = datasets.ImageFolder(TRAIN_DIR)
targets = raw_dataset.targets

train_idx, val_idx = train_test_split(
	list(range(len(targets))),
	test_size=0.1,
	stratify=targets,
	random_state=42
)

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

train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
val_dataset = datasets.ImageFolder(TRAIN_DIR, transform=val_tf)

train_set = Subset(train_dataset, train_idx)
val_set = Subset(val_dataset, val_idx)

train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=4)
val_loader = DataLoader(val_set, batch_size=16, shuffle=False, num_workers=4)

# --- MODEL ---
# weights = EfficientNet_V2_M_Weights.DEFAULT
# model = efficientnet_v2_m(weights=weights)
weights = EfficientNet_V2_L_Weights.DEFAULT
model = efficientnet_v2_l(weights=weights)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model = model.to(device)

# --- OPTIMIZER ---
optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-2)
criterion = nn.CrossEntropyLoss()

# --- SCHEDULER : WARMUP + COSINE ---
num_epochs = 50
warmup_epochs = 5
cosine_epochs = num_epochs - warmup_epochs

def lr_lambda(epoch):
	if epoch < warmup_epochs:
		return (epoch + 1) / warmup_epochs
	progress = (epoch - warmup_epochs) / cosine_epochs
	return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.1415926535)))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# --- EARLY STOPPING ---
best_val_loss = float('inf')
patience = 5
counter = 0
best_model_wts = copy.deepcopy(model.state_dict())

# --- TRAINING LOOP ---
for epoch in range(num_epochs):
	torch.cuda.empty_cache()

	# ---------------- TRAIN ----------------
	model.train()
	total_loss, total, correct = 0, 0, 0

	for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch} Train"):
		imgs, labels = imgs.to(device), labels.to(device)

		optimizer.zero_grad()
		logits = model(imgs)
		loss = criterion(logits, labels)
		loss.backward()
		optimizer.step()

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
			logits = model(imgs)
			loss = criterion(logits, labels)

			val_loss_sum += loss.item() * imgs.size(0)
			val_correct += (logits.argmax(1) == labels).sum().item()
			val_total += imgs.size(0)

	val_acc = val_correct / val_total
	val_loss = val_loss_sum / val_total

	print(f"Epoch {epoch}: Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Val Loss: {val_loss:.4f}")

	# ---------------- SCHEDULER UPDATE ----------------
	scheduler.step()

	# ---------------- EARLY STOPPING ----------------
	if val_loss < best_val_loss:
		best_val_loss = val_loss
		best_model_wts = copy.deepcopy(model.state_dict())
		counter = 0
		torch.save(best_model_wts, "/scratch/Shawn/best_modelL.pth")
	else:
		counter += 1
		print(f"EarlyStopping counter: {counter}/{patience}")
		if counter >= patience:
			print("Early stopping triggered!")
			break

	# ---------------- LEADERBOARD SUBMIT ----------------
	current_lr = float(optimizer.param_groups[0]['lr'])

	lb.submit_training(
		train_acc,
		train_loss,
		{
			"num_epochs": num_epochs,
			"lr": current_lr,
			"batch_size": 16,
			"epoch": epoch
		},
		tag=f"EffNetV2L : epoch:{epoch}, lr:{current_lr:.6f}, batch_size:16"
	)

print("Fin de l'entraînement.")
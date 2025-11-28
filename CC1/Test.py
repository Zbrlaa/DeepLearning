import os
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import leaderboard_client as lb
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights
from tqdm import tqdm
from PIL import Image

#Config
ROOT = "/scratch/Shawn/download/FCUNIST"
TRAIN_DIR = os.path.join(ROOT, "train")
TEST_DIR = os.path.join(ROOT, "test")
os.environ['LEADERBOARD_TOKEN'] = 'e99def1e-e708-4c32-8591-78d6d27cd317'

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

test_tf = transforms.Compose([
	transforms.Resize((224, 224)),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406],
						std=[0.229, 0.224, 0.225]),
])


#Datasets
train_set = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)

num_classes = len(os.listdir(TRAIN_DIR))
print(f"{num_classes} classes")

test_files = sorted(os.listdir(TEST_DIR))

class TestDataset(Dataset):
    def __init__(self, root, tf):
        self.root = root
        self.files = sorted(os.listdir(root))
        self.tf = tf

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        path = os.path.join(self.root, filename)
        img = Image.open(path).convert("RGB")
        return self.tf(img), filename

test_loader = DataLoader(TestDataset(TEST_DIR, test_tf), batch_size=32, shuffle=False, num_workers=4)


#Model
weights = EfficientNet_V2_M_Weights.DEFAULT
model = efficientnet_v2_m(weights=weights)
 
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

model = model.to(device)

#Optim
lr = 3e-4
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
criterion = nn.CrossEntropyLoss()

num_epochs = 10


#Train
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

	print(f"Epoch {epoch}:")
	print(f"  Train     - acc={train_acc:.4f} loss={train_loss:.4f}")
	print("-----------------------------------------------------")

	lb.submit_training(
		train_acc,
		train_loss,
		{
			"num_epochs": num_epochs,
			"lr": lr,
			"batch_size": 32,
			"epoch": epoch
		},
		tag=f"EffNetV2M : epoch:{epoch}, lr:{lr}, batch_size:32"
	)

#Test
model.eval()
all_preds = {}
class_name = train_set.classes

with torch.no_grad():
    for imgs, filenames in tqdm(test_loader, desc="Test"):
        imgs = imgs.to(device)

        logits = model(imgs).cpu()

        for file, logit_vec in zip(filenames, logits):
            all_preds[file] = {
                cls: float(logit_vec[i])
                for i, cls in enumerate(class_name)
            }

print(all_preds)
response = lb.submit_test(all_preds)
print(response)
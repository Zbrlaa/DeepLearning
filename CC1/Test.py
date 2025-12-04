import os
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import leaderboard_client as lb
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
# from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights
from torchvision.models import efficientnet_v2_l, EfficientNet_V2_L_Weights
from tqdm import tqdm
from PIL import Image

#Config
ROOT = "/scratch/Shawn/download/FCUNIST"
TRAIN_DIR = os.path.join(ROOT, "train")
TEST_DIR = os.path.join(ROOT, "test")
os.environ['LEADERBOARD_TOKEN'] = 'e99def1e-e708-4c32-8591-78d6d27cd317'
os.environ['TORCH_HOME'] = '/scratch/Shawn/.cache/torch'

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
# weights = EfficientNet_V2_M_Weights.DEFAULT
# model = efficientnet_v2_m(weights=weights)
weights = EfficientNet_V2_L_Weights.DEFAULT
model = efficientnet_v2_l(weights=weights)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model = model.to(device)

# --- LOAD BEST MODEL ---
# Charger ton modèle optimisé (entraîné dans l’autre script)
model.load_state_dict(torch.load("/scratch/Shawn/best_modelL.pth", map_location=device))
model.eval()
print(">>> Chargement du meilleur modèle effectué.")

#Test
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

# Envoi au leaderboard
response = lb.submit_test(all_preds)
print(response)
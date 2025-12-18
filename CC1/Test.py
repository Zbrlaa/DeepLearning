import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.models import efficientnet_v2_l, EfficientNet_V2_L_Weights
from PIL import Image
from tqdm import tqdm
import leaderboard_client as lb

# ---------------- CONFIG ----------------
ROOT = "/scratch/Shawn/download/FCUNIST"
TRAIN_DIR = os.path.join(ROOT, "train")
TEST_DIR = os.path.join(ROOT, "test")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

os.environ['LEADERBOARD_TOKEN'] = 'e99def1e-e708-4c32-8591-78d6d27cd317'
os.environ['TORCH_HOME'] = '/scratch/Shawn/.cache/torch'

# ---------------- TRANSFORMS ----------------
weights = EfficientNet_V2_L_Weights.DEFAULT
base_tf = weights.transforms()   # Pour récupérer mean/std corrects

# EXACTEMENT LES MÊMES QUE DANS LE TRAIN
test_tf = transforms.Compose([
	transforms.Resize(232),
	transforms.CenterCrop(224),
	transforms.ToTensor(),
	transforms.Normalize(mean=base_tf.mean, std=base_tf.std)
])

# ---------------- DATASET TEST ----------------
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


test_loader = DataLoader(TestDataset(TEST_DIR, test_tf),
						batch_size=32,
						shuffle=False,
						num_workers=4)

# ---------------- MODEL ----------------
num_classes = len(os.listdir(TRAIN_DIR))
print(f"{num_classes} classes")

model = efficientnet_v2_l(weights=weights)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model = model.to(device)

# LOAD BEST MODEL FROM TRAIN SCRIPT
model.load_state_dict(torch.load("/scratch/Shawn/best_modelL.pth", map_location=device))
model.eval()
print(">>> Meilleur modèle chargé avec succès.")

# ---------------- TEST INFERENCE ----------------
all_preds = {}
class_names = sorted(os.listdir(TRAIN_DIR))

with torch.no_grad():
	for imgs, filenames in tqdm(test_loader, desc="Test"):
		imgs = imgs.to(device)

		logits = model(imgs).cpu()

		for file, logit_vec in zip(filenames, logits):
			all_preds[file] = {
				cls: float(logit_vec[i])
				for i, cls in enumerate(class_names)
			}

# ---------------- SUBMIT LEADERBOARD ----------------
response = lb.submit_test(all_preds)
print(response)
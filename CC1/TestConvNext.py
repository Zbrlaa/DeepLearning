import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import convnext_small, ConvNeXt_Small_Weights
from PIL import Image
import leaderboard_client as lb
from tqdm import tqdm

# ---------------- CONFIG ----------------
ROOT = "/scratch/Shawn/download/FCUNIST"
TEST_DIR = os.path.join(ROOT, "test")
os.environ['LEADERBOARD_TOKEN'] = 'e99def1e-e708-4c32-8591-78d6d27cd317'
os.environ['TORCH_HOME'] = '/scratch/Shawn/.cache/torch'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ---------------- WEIGHTS ----------------
weights = ConvNeXt_Small_Weights.IMAGENET1K_V1
base_tf = weights.transforms()

# ---------------- TRANSFORMATIONS ----------------
test_tf = transforms.Compose([
	transforms.Resize(232),
	transforms.CenterCrop(224),
	transforms.ToTensor(),
	transforms.Normalize(mean=base_tf.mean, std=base_tf.std)
])

# ---------------- MODEL ----------------
num_classes = len(os.listdir(os.path.join(ROOT, "train")))
model = convnext_small(weights=weights)
model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features, num_classes)
model.load_state_dict(torch.load("/scratch/Shawn/best_model_convnext_small.pth", map_location=device))
model = model.to(device)
model.eval()
print(">>> Chargement du meilleur modèle effectué.")

# ---------------- DATASET ----------------
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

# ---------------- INFERENCE ----------------
all_preds = {}
class_name = sorted(os.listdir(os.path.join(ROOT, "train")))

with torch.no_grad():
	for imgs, filenames in tqdm(test_loader, desc="Test"):
		imgs = imgs.to(device)
		logits = model(imgs).cpu()
		for file, logit_vec in zip(filenames, logits):
			all_preds[file] = {cls: float(logit_vec[i]) for i, cls in enumerate(class_name)}

# ---------------- LEADERBOARD ----------------
response = lb.submit_test(all_preds)
print(response)
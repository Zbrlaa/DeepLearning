import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import Imagenette, ImageFolder
import torchvision.transforms as TF
from torch.optim import AdamW
from tqdm import tqdm


device = torch.device("cuda")

class Fire(nn.Module):
	def __init__(self, in_channels, squeeze_channels, expand_channels):
		super(Fire, self).__init__()

		self.relu = nn.ReLU(inplace=True)

		self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)

		self.expand1 = nn.Conv2d(squeeze_channels, expand_channels, kernel_size=1)
		self.expand3 = nn.Conv2d(squeeze_channels, expand_channels, kernel_size=3, padding=1)

	def forward(self, x):
		x = self.squeeze(x)
		x = self.relu(x)
		x1 = self.expand1(x)
		x1 = self.relu(x1)
		x2 = self.expand3(x)
		x2 = self.relu(x2)
		x = torch.concat([x1,x2], dim=1)
		return x


class SqueezeNet(nn.Module):
	def __init__(self, num_class=10):
		super(SqueezeNet, self).__init__()

		self.trunk = nn.Sequential(
			nn.Conv2d(3, 96, kernel_size=7, stride=2),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2),

			Fire(96, 16, 64),
			Fire(128, 16, 64),
			Fire(128, 32, 128),
			nn.MaxPool2d(kernel_size=3, stride=2),

			Fire(256, 32, 128),
			Fire(256, 48, 192),
			Fire(384, 48, 192),
			Fire(384, 64, 256),
			nn.MaxPool2d(kernel_size=3, stride=2),

			Fire(512, 64, 256),
			# nn.Dropout2d(p=0.5),
			nn.Conv2d(512, 10, kernel_size=1, stride=1),
			nn.AdaptiveAvgPool2d(1)
		)


	def forward(self, x):
		features = self.trunk(x)
		B, C, H, W = features.shape
		return features.view(B, -1)


train_tfm = TF.Compose(
		[
			TF.Resize(256),
			TF.CenterCrop(224),
			TF.ToTensor(),
			TF.Normalize(mean=[0.485, 0.456, 0.406],
				std=[0.229, 0.224, 0.225])
		]
	)

train_set = Imagenette(
	"imagenette", download=True, size="320px", split="train", transform=train_tfm
)
val_set = Imagenette(
	"imagenette", download=True, size="320px", split="val", transform=train_tfm
)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=256, shuffle=False)

model = SqueezeNet().to(device)
optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)


def train(model, loader):
	model.train()
	running_loss = 0
	correct = 0
	total = 0

	for imgs, labels in tqdm(loader, desc="Training") :
		imgs, labels = imgs.to(device), labels.to(device)

		optimizer.zero_grad()
		outputs = model(imgs)
		loss = F.cross_entropy(outputs, labels)
		loss.backward()
		optimizer.step()

		running_loss += loss.item()
		_, predicted = outputs.max(1)
		total += labels.size(0)
		correct += predicted.eq(labels).sum().item()

	return running_loss / len(loader), 100 * correct / total


def evaluate(model, loader):
	model.eval()
	loss = 0
	correct = 0
	total = 0

	with torch.no_grad():
		for imgs, labels in loader:
			imgs, labels = imgs.to(device), labels.to(device)
			outputs = model(imgs)
			loss += F.cross_entropy(outputs, labels).item()
			_, predicted = outputs.max(1)
			total += labels.size(0)
			correct += predicted.eq(labels).sum().item()

	return loss / len(loader), 100 * correct / total


EPOCHS = 10

for epoch in range(EPOCHS):
	print(f"Epoch {epoch+1}/{EPOCHS}")
	train_loss, train_acc = train(model, train_loader)
	val_loss, val_acc = evaluate(model, val_loader)

	print(f" Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
	print(f" Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%\n")
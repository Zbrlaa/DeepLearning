import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torchvision.datasets import Imagenette
import torchvision.transforms as TF
from visdom import Visdom
import matplotlib.pyplot as plt

device = "cpu"

vis = Visdom(env="lesson")



def get_model() :
	return nn.Sequential(
		nn.Conv2d(3,64, kernel_size=3, padding=1),
		nn.ReLU(),
		nn.Conv2d(64, 64, kernel_size=3, padding=1),
		nn.ReLU(),
		nn.MaxPool()

	)

def main() :
	epochs = 10
	vis.close()
	train_tfm = TF.Compose([
		TF.Resize(160),
		TF.Crop(160),
		TF.ToTensor(),
		TF.Normalize([0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
	])
	
	train_set = Imagenette('imagenette', download=True, size="160px", split='train', transform=tfm)
	print(train_set.classes[train_set[0][1]])
	vis.image(train_set[0][0], win="demo")
	test_set = Imagenette('imagenette', download=True, size="160px", split='val', transform=tfm)

	model = get_model()
	model.to(device)

	opt = AdamW(model.parameters(), lr=3e-4)

	train_loader = DataLoader(train_set, batch_size=32)
	for _ in range(epochs) :
		for x, y in train_loader :
			x = x.to(device)
			y = y.to(device)
			opt.zero_grad()
			pred = model(x)
			loss = F.cross_entropy(pred, y)
			loss.backward()
			print(loss.item())
			opt.step()

if __name__ == '__main__ ':
	main()
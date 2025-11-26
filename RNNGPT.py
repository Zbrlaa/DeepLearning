import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda")

class MyRNN(nn.Module):
	"""
	h = tanh(Wh @ h + Wx @ x)
	o = Wo @ h
	"""

	def __init__(self, in_dim, h_dim, out_dim) -> None:
		super().__init__()
		self.h_dim = h_dim
		self.Wh = nn.Linear(h_dim, h_dim, bias=False)
		self.Wx = nn.Linear(in_dim, h_dim, bias=False)
		self.Wo = nn.Linear(h_dim, out_dim, bias=False)

	def forward(self, xs):
		# xs: [B, L, D]
		B, L, D = xs.shape
		H = self.h_dim
		outs = []

		# IMPORTANT : hidden state sur le bon device
		h = torch.zeros(B, H, device=xs.device)

		for x in xs.transpose(0, 1):  # BLD -> LBD
			# x: [B, D]
			h = torch.tanh(self.Wh(h) + self.Wx(x))  # BH
			o = self.Wo(h)  # BH
			outs.append(o)

		return torch.stack(outs, dim=1)  # list(BO) -> BLO


class MyCharRNN(nn.Module):
	def __init__(self):
		super().__init__()
		self.embed = nn.Embedding(256, 64)
		self.rnn = MyRNN(64, 256, 256)

	def forward(self, xs):
		return self.rnn(self.embed(xs))


# --------------------------
# Chargement dataset
# --------------------------

with open("CodeGeassScript.txt", "r", encoding="utf-8") as f:
	dataset = f.read()

dataset = dataset.encode("ascii", errors="replace").decode("ascii")
data = [ord(c) for c in dataset]
data = data[: int(len(data) / 129) * 129]
data = torch.tensor(data, dtype=torch.long).view(-1, 129)


# --------------------------
# Entraînement
# --------------------------

def train():
	model = MyCharRNN().to(device)
	opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
	epochs = 100
	dl = torch.utils.data.DataLoader(data, batch_size=8, shuffle=True)

	generate(model)

	for epoch in range(epochs):
		for batch in dl:
			batch = batch.to(device)  # batch → CUDA
			opt.zero_grad()

			pred = model(batch[:, :-1])
			loss = F.cross_entropy(pred.transpose(1, 2), batch[:, 1:])
			loss.backward()
			opt.step()

		print(f"Epoque {epoch}")
		generate(model)

	return model


# --------------------------
# Génération
# --------------------------

def generate(model):
	generated = "J"
	for _ in range(128):
		# IMPORTANT : mettre sur device
		gen_idx = torch.tensor([ord(c) for c in generated], dtype=torch.long, device=device)

		pred = model(gen_idx.unsqueeze(0))  # [1, L, vocab]
		pred = pred[:, -1, :]               # dernier token

		next_idx = torch.multinomial(F.softmax(pred, dim=-1), num_samples=1).squeeze(1)

		generated += chr(next_idx.item())

	print(generated)


# --------------------------
# Main
# --------------------------

if __name__ == "__main__":
	model = train()
	generate(model)
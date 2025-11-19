import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW

# -----------------------------
# RNN “from scratch”
# -----------------------------
class MyRNN(nn.Module):
	"""
	h_t = tanh(Wh @ h_{t-1} + Wx @ x_t)
	o_t = Wo @ h_t
	"""
	def __init__(self, in_dim, h_dim, out_dim):
		super().__init__()
		self.h_dim = h_dim
		self.Wh = nn.Linear(h_dim, h_dim, bias=False)
		self.Wx = nn.Linear(in_dim, h_dim, bias=False)
		self.Wo = nn.Linear(h_dim, out_dim, bias=False)

	def forward(self, xs):
		# xs: [B, L, D]
		B, L, D = xs.shape
		H = self.h_dim
		h = torch.zeros(B, H, device=xs.device)
		outs = []

		for x in xs.transpose(0, 1):  # boucle sur la séquence
			h = torch.tanh(self.Wh(h) + self.Wx(x))
			o = self.Wo(h)
			outs.append(o)

		return torch.stack(outs, dim=1)  # [B, L, out_dim]

# -----------------------------
# Char-level RNN
# -----------------------------
class MyCharRNN(nn.Module):
	def __init__(self, vocab_size=256, embed_dim=64, hidden_dim=256):
		super().__init__()
		self.embed = nn.Embedding(vocab_size, embed_dim)
		self.rnn = MyRNN(embed_dim, hidden_dim, vocab_size)

	def forward(self, xs):
		# xs: [B, L] d'indices de caractères
		return self.rnn(self.embed(xs))

# -----------------------------
# Préparation du texte
# -----------------------------
with open("CodeGeassScript.txt", "r", encoding="utf-8") as f:
	dataset = f.read()

# convertir chaque caractère en index ASCII
data = [ord(c) for c in dataset]
# tronquer pour que la longueur soit divisible par la séquence
seq_len = 129
data = data[:len(data) // seq_len * seq_len]
data = torch.tensor(data, dtype=torch.long)
# former des séquences de longueur seq_len
data = data.view(-1, seq_len)

# Détection du device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# Boucle d’entraînement
# -----------------------------
def train():
	model = MyCharRNN().to(device)  # déplacer le modèle sur le GPU
	opt = AdamW(model.parameters(), lr=3e-4)
	epochs = 10
	dl = DataLoader(data, batch_size=8, shuffle=True)

	for _epoch in range(epochs):
		for batch in dl:
			batch = batch.to(device)          # déplacer le batch sur le GPU
			opt.zero_grad()
			pred = model(batch[:, :-1])
			loss = F.cross_entropy(pred.transpose(1, 2), batch[:, 1:])
			loss.backward()
			opt.step()
		print(f"Epoch {_epoch+1}/{epochs}, Loss: {loss.item():.4f}")

	return model

# -----------------------------
# Génération de texte
# -----------------------------
def generate(model, start="L", length=32):
	generated = start
	model.eval()
	
	for _ in range(length):
		gen_idx = torch.tensor([ord(c) for c in generated], dtype=torch.long, device=device)
		pred = model(gen_idx.unsqueeze(0))
		pred_last = pred[:, -1, :]
		next_idx = pred_last.argmax(dim=1)
		generated += chr(next_idx.item())

	print("Generated text:\n", generated)

# -----------------------------
# Entraîner et générer
# -----------------------------
model = train()
generate(model, start="L")
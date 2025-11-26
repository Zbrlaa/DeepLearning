import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda")

class MyMLP(nn.Module) :
	def __init__(self, dim) :
		super().__init__()
		self.linear1 = nn.Linear(dim, 4*dim)
		self.linear2 = nn.Linear(4*dim, dim)

	def forward(self, x) :
		x = self.linear1(x)
		x = F.gelu(x)
		x = self.linear2(x)
		return x

class MySelfAttention(nn.Module) :
	def __init__(self, dim, num_heads=8) :
		super().__init__()
		self.q = nn.Linear(dim, dim)
		self.k = nn.Linear(dim, dim)
		self.v = nn.Linear(dim, dim)
		self.o = nn.Linear(dim, dim)
		self.num_heads = num_heads

	def forward(self, x) :
		# x : BLD
		q = self.q(x)
		k = self.k(x)
		v = self.v(x)
		# k, q, v : BLD
		B, L, D = k.shape
		q = q.view(B, L, self.num_heads, D // self.num_heads)
		k = k.view(B, L, self.num_heads, D // self.num_heads)
		v = v.view(B, L, self.num_heads, D // self.num_heads)

		#k : BLHk
		q = q.permute(0, 2, 1, 3) #BHLk
		k = k.permute(0, 2, 1, 3) #BHLk
		v = v.permute(0, 2, 1, 3) #BHLk

		out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
		# out : BHLk
		out = out.permute(0, 2, 1, 3)
		# out : BLHk
		out = out.view(B, L, D)
		# out : BL(H K)
		
		o = self.o(out)
		return o

class Block(nn.Module) :
	def __init__(self, dim) :
		super().__init__()
		self.rms = nn.RMSNorm(dim)
		self.self_attention = MySelfAttention(dim)
		self.mlp = MyMLP(dim)

	def forward(self, x) :
		x = x + self.self_attention(self.rms(x))
		x = x + self.mlp(self.rms(x))
		return x


class OptimusPrime(nn.Module):
	def __init__(self, vocab_size, dim, num_blocks) :
		super().__init__()
		self.emb = nn.Embedding(vocab_size, dim)
		self.blocks = nn.ModuleList([
			Block(dim) for _ in range(num_blocks)
		])
		self.final_rms = nn.RMSNorm(dim)
		self.output_layer = nn.Linear(dim, vocab_size)
	
	def forward(self, x) :
		# x : [B, L]
		emb = self.emb(x)
		# emb : [B, L, D]
		for block in self.blocks :
			emb = block(emb)
			#emb : [B, L, D]
		emb = self.final_rms(emb)
		# emb : [B, L, D]
		out = self.output_layer(emb)
		# out : [B, L, D_out]
		return out

def main() :
	print("Lancement test")
	B = 4           # batch size
	L = 6           # longueur séquence
	vocab_size = 50
	dim = 32
	num_blocks = 2
	lr = 1e-3
	num_steps = 100

	model = OptimusPrime(vocab_size, dim, num_blocks).to(device)
	optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
	criterion = nn.CrossEntropyLoss()

	# X : tokens aléatoires
	# Y : tokens décalés de 1 (next token prediction)
	for step in range(num_steps):
		x = torch.randint(0, vocab_size, (B, L)).to(device)
		y = torch.randint(0, vocab_size, (B, L)).to(device)  # target factice

		optimizer.zero_grad()
		out = model(x)               # [B, L, vocab_size]
		loss = criterion(out.view(-1, vocab_size), y.view(-1))
		loss.backward()
		optimizer.step()

		if step % 10 == 0:
			print(f"Step {step} - Loss: {loss.item():.4f}")

main()
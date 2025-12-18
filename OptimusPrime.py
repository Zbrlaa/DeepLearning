import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import leaderboard as lb
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import random_split
from tqdm import tqdm

os.environ['LEADERBOARD_TOKEN'] = '7085ed97-e16f-4821-bc0d-172ab46e9298'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MyMLP(nn.Module) :
	def __init__(self, dim, dropout=0.1) :
		super().__init__()
		self.linear1 = nn.Linear(dim, 4*dim)
		self.linear2 = nn.Linear(4*dim, dim)
		self.dropout = nn.Dropout(dropout) # Recommandé pour éviter l'overfitting

	def forward(self, x) :
		x = self.linear1(x)
		x = F.gelu(x)
		x = self.linear2(x)
		return self.dropout(x)

class MySelfAttention(nn.Module) :
	def __init__(self, dim, num_heads=8, dropout=0.1) :
		super().__init__()
		self.q = nn.Linear(dim, dim)
		self.k = nn.Linear(dim, dim)
		self.v = nn.Linear(dim, dim)
		self.o = nn.Linear(dim, dim)
		self.num_heads = num_heads
		self.dropout = nn.Dropout(dropout)

	def forward(self, x) :
		B, L, D = x.shape
		q = self.q(x).view(B, L, self.num_heads, D // self.num_heads).permute(0, 2, 1, 3)
		k = self.k(x).view(B, L, self.num_heads, D // self.num_heads).permute(0, 2, 1, 3)
		v = self.v(x).view(B, L, self.num_heads, D // self.num_heads).permute(0, 2, 1, 3)

		# Utilisation du dropout intégré à l'attention de PyTorch
		out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout.p if self.training else 0.0)
		
		out = out.permute(0, 2, 1, 3).contiguous().view(B, L, D)
		return self.dropout(self.o(out))

class Block(nn.Module) :
	def __init__(self, dim, num_heads=8, dropout=0.1) :
		super().__init__()
		self.rms = nn.RMSNorm(dim)
		self.self_attention = MySelfAttention(dim, num_heads, dropout)
		self.mlp = MyMLP(dim, dropout)

	def forward(self, x) :
		# Pre-norm architecture
		x = x + self.self_attention(self.rms(x))
		x = x + self.mlp(self.rms(x))
		return x

class OptimusPrime(nn.Module):
	def __init__(self, vocab_size=1025, dim=512, num_blocks=8, max_len=257, num_heads=8, dropout=0.1) :
		super().__init__()
		self.max_len = max_len
		self.emb = nn.Embedding(vocab_size, dim)
		
		# 1. POSITION ENCODING (Learned)
		# On crée un paramètre apprenable pour chaque position de 0 à 256
		self.pos_emb = nn.Parameter(torch.zeros(1, max_len, dim))
		
		self.blocks = nn.ModuleList([
			Block(dim, num_heads, dropout) for _ in range(num_blocks)
		])
		self.final_rms = nn.RMSNorm(dim)
		
		# 2. WEIGHT TYING (Liaison des poids)
		# On utilise bias=False pour que les dimensions correspondent parfaitement
		self.output_layer = nn.Linear(dim, vocab_size, bias=False)
		self.output_layer.weight = self.emb.weight 
		
		# Initialisation propre des poids (crucial pour les Transformers)
		self.apply(self._init_weights)

	def _init_weights(self, module):
		if isinstance(module, nn.Linear):
			torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
		elif isinstance(module, nn.Embedding):
			torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

	def forward(self, x) :
		B, L = x.shape
		# Somme de l'embedding de token et de position
		x = self.emb(x) + self.pos_emb[:, :L, :]
		
		for block in self.blocks :
			x = block(x)
			
		x = self.final_rms(x)
		return self.output_layer(x)

	@torch.no_grad()
	def generate(self, idx, max_new_tokens, temperature=1.0):
		# Pour prédire la suite d'une image au test set
		for _ in range(max_new_tokens):
			idx_cond = idx[:, -self.max_len:]
			logits = self(idx_cond)
			logits = logits[:, -1, :] / temperature
			probs = F.softmax(logits, dim=-1)
			idx_next = torch.multinomial(probs, num_samples=1)
			idx = torch.cat((idx, idx_next), dim=1)
			if idx.shape[1] >= self.max_len: break
		return idx


class MythosDataset(Dataset):
	def __init__(self, file_path):
		self.data = []
		with open(file_path, 'r') as f:
			for line in f:
				parts = line.strip().split()
				if len(parts) > 1:
					# On ignore le premier élément (le tag)
					tokens = [int(t) for t in parts[1:]]
					self.data.append(torch.tensor(tokens))

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		tokens = self.data[idx]
		# X : les 256 premiers tokens
		# Y : les tokens décalés (du 2ème au 257ème)
		return tokens[:-1], tokens[1:]


def get_scheduler(optimizer, warmup_steps, total_steps):
	# 1. Phase de Warmup : augmente le LR de 0.1*lr à 1.0*lr
	scheduler_warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
	
	# 2. Phase de Cosine : descend de 1.0*lr à 0
	scheduler_cosine = CosineAnnealingLR(optimizer, T_max=(total_steps - warmup_steps))
	
	# Combine les deux
	return SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[warmup_steps])


def validate(model, val_loader, criterion):
    model.eval() # Mode évaluation (désactive le dropout)
    total_loss = 0
    
    with torch.no_grad(): # Pas de calcul de gradient (gain de mémoire/vitesse)
        for x, y in tqdm(val_loader, desc="Validation"):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits.view(-1, 1025), y.view(-1))
            total_loss += loss.item()
            
    avg_loss = total_loss / len(val_loader)
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity


def main():
	# 1. Hyperparamètres (Vise ces valeurs pour de bonnes perfs)
	vocab_size = 1025
	dim = 512
	num_blocks = 8
	num_heads = 8
	max_len = 257  # Séquence totale
	batch_size = 64
	lr = 5e-4      # LR standard pour cette taille de modèle
	num_epochs = 15
	
	# 2. Préparation
	model = OptimusPrime(vocab_size=vocab_size, dim=dim, num_blocks=num_blocks, max_len=max_len, num_heads=num_heads).to(device)
	dataset = MythosDataset("dataset.txt")

	train_size = int(0.9 * len(dataset))
	val_size = len(dataset) - train_size
	train_data, val_data = random_split(dataset, [train_size, val_size])

	train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
	val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
	
	optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
	
	warmup_steps = 500
	total_steps = len(train_loader) * num_epochs
	scheduler = get_scheduler(optimizer, warmup_steps, total_steps)
	
	criterion = nn.CrossEntropyLoss()

	for epoch in range(num_epochs):
		model.train()
		total_loss = 0
		total_acc = 0
		
		pbar = tqdm(train_loader, desc=f"Train E{epoch}")
		for x, y in pbar:
			x, y = x.to(device), y.to(device)
			
			optimizer.zero_grad()
			logits = model(x)
			
			# Calcul de la Loss
			loss = criterion(logits.view(-1, 1025), y.view(-1))
			loss.backward()
			
			# Gradient clipping pour la stabilité
			torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
			
			optimizer.step()
			scheduler.step() # On update le LR à chaque batch
			
			# Calcul de l'accuracy (pour le leaderboard)
			with torch.no_grad():
				preds = torch.argmax(logits, dim=-1)
				acc = (preds == y).float().mean()
			
			total_loss += loss.item()
			total_acc += acc.item()
			current_lr = float(optimizer.param_groups[0]['lr'])

			pbar.set_postfix(loss=loss.item(), acc=acc.item(), lr=f"{current_lr:.2e}")

		# Calcul des moyennes de l'epoch
		epoch_loss = total_loss / len(train_loader)
		epoch_acc = total_acc / len(train_loader)

		val_loss, val_ppx = validate(model, val_loader, criterion)
		print(f"Epoch {epoch} - Val Loss: {val_loss:.4f} - Perplexity: {val_ppx:.4f}")

		# --- SOUMISSION AU LEADERBOARD ---
		# On envoie les metrics pour cette epoch
		print(f"Epoch {epoch} terminée. Soumission...")
		current_lr = float(optimizer.param_groups[0]['lr'])
		lb.submit_training(
			accuracy=epoch_acc, 
			loss=epoch_loss, 
			hyperparameters={"lr": current_lr, "batch_size": batch_size, "warmup": warmup_steps},
			tag=f"OptimusPrime_v1_epoch_{epoch}"
		)

	# Sauvegarde ton modèle !
	torch.save(model.state_dict(), "/scratch/Shawn/model_mythos.pth")

if __name__ == "__main__":
    main()
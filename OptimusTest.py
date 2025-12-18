import torch
import os
import leaderboard as lb
from OptimusPrime import OptimusPrime
from tqdm import tqdm

os.environ['LEADERBOARD_TOKEN'] = '7085ed97-e16f-4821-bc0d-172ab46e9298'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_test_submission(model_path, test_file="dataset-test.txt"):
	vocab_size = 1025
	dim = 512
	num_blocks = 8
	num_heads = 8
	max_len = 257
	
	model = OptimusPrime(
		vocab_size=vocab_size, 
		dim=dim, 
		num_blocks=num_blocks, 
		max_len=max_len, 
		num_heads=num_heads
	).to(device)
	
	# Charger les poids sauvegardés
	print(f"Chargement du modèle : {model_path}")
	model.load_state_dict(torch.load(model_path, map_location=device))
	model.eval()

	all_predictions = {}

	if not os.path.exists(test_file):
		print(f"Erreur : Le fichier {test_file} est introuvable.")
		return

	print(f"Génération des séquences à partir de {test_file}...")
	with open(test_file, 'r') as f:
		lines = f.readlines()

	for line in tqdm(lines, desc="Test"):
		parts = line.strip().split()
		if not parts:
			continue
		
		tag = parts[0]
		# Conversion des tokens du préfixe en entiers
		prefix_tokens = [int(t) for t in parts[1:]]
		
		# Préparation du tenseur d'entrée [Batch=1, Longueur_Prefixe]
		x = torch.tensor([prefix_tokens], dtype=torch.long).to(device)
		
		# Calcul du nombre de tokens à générer pour atteindre 257 au total
		tokens_to_generate = max_len - len(prefix_tokens)
		
		if tokens_to_generate > 0:
			with torch.no_grad():
				# On utilise ta fonction generate avec une température de 0.8
				# 0.8 est un bon compromis pour la cohérence visuelle
				generated_seq = model.generate(x, max_new_tokens=tokens_to_generate, temperature=0.8)
				final_tokens = generated_seq[0].tolist()
		else:
			final_tokens = prefix_tokens[:max_len]
		
		# Stockage dans le dictionnaire final
		all_predictions[tag] = final_tokens

	# 3. Soumission au leaderboard
	print("Envoi des prédictions au serveur...")
	try:
		response = lb.submit_test(all_predictions)
		print("Réponse du serveur :", response)
	except Exception as e:
		print(f"Erreur lors de la soumission : {e}")

if __name__ == "__main__":
	run_test_submission("/scratch/Shawn/model_mythos.pth")
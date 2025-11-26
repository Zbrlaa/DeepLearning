import os
os.environ['LEADERBOARD_TOKEN'] = 'YOUR_TOKEN'
import leaderboard_client as lb

lb.submit_training(train_acc, train_loss, {"num_epochs": num_epochs, "lr": lr, "something_else": 12}, tag=f"super net epoch {epoch}")
all_preds = {"file1.png": {"storm": -3.32, "doctorstrange": 0.2, ...}, ...}
lb.submit_test(all_preds)
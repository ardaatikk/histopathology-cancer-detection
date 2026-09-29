import pandas as pd
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser(description='')
parser.add_argument('--log_file', default='', help='Log file')

FLAGS = parser.parse_args()

log_file = FLAGS.log_file

log_df = pd.read_csv(log_file, delimiter=",", header=0)

# Extract the values for loss and accuracy
train_losses = log_df["Train_Loss"].astype(float)  
train_accuracies = log_df["Train_Accuracy"].astype(float) 
valid_losses = log_df["Valid_Loss"].astype(float) 
valid_accuracies = log_df["Valid_Accuracy"].astype(float) 

# train_losses = log_df[" Train Loss"].astype(float)  
# train_accuracies = log_df[" Train Accuracy"].str[:-1].astype(float) 
# valid_losses = log_df[" Valid Loss"].astype(float) 
# valid_accuracies = log_df[" Valid Accuracy"].str[:-1].astype(float) 

# Create a figure with subplots
fig, axs = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(f'{log_file}Loss and Accuracy Curves')

# Plot the loss curves
axs[0].plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
axs[0].plot(range(1, len(valid_losses) + 1), valid_losses, label="Valid Loss")
axs[0].set_xlabel("Epoch")
axs[0].set_ylabel("Loss")
axs[0].set_title("Training and Validation Loss")
axs[0].legend()

# Plot the accuracy curves
axs[1].plot(range(1, len(train_accuracies) + 1), train_accuracies, label="Train Accuracy")
axs[1].plot(range(1, len(valid_accuracies) + 1), valid_accuracies, label="Valid Accuracy")
axs[1].set_xlabel("Epoch")
axs[1].set_ylabel("Accuracy (%)")
axs[1].set_title("Training and Validation Accuracy")
axs[1].legend()

# Adjust spacing between subplots
plt.tight_layout()
plt.savefig(f"{log_file}_plots.pdf")
plt.savefig(f"{log_file}_plots.png")
# Show the plot
plt.show()

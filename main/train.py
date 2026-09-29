import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import datetime
import argparse

from torch.utils.data import DataLoader
from model import MyModel
from dataset import Dataset

parser = argparse.ArgumentParser(description='')
parser.add_argument('--img_dir', default='./images/images', help='Images directory')
parser.add_argument('--train_file', default='./myTrain.csv', help='Train File')
parser.add_argument('--valid_file', default='./myValid.csv', help='Validation file')
parser.add_argument('--learning_rate', default=0.001, help='Learning rate')
parser.add_argument('--batch_size', default=16, help='Batch size')
parser.add_argument('--num_epochs', default=20, help='Number of epochs')
parser.add_argument('--momentum', default=0.9, help='Momentum')

FLAGS = parser.parse_args()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define hyperparameters
learning_rate = FLAGS.learning_rate
batch_size = FLAGS.batch_size
num_epochs = FLAGS.num_epochs
momentum = FLAGS.momentum
# torch.manual_seed(13)

best_valid_loss = float(10000.000)

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)), # Resize the images to 224 x 224 pixels
    transforms.Normalize((0.62376275, 0.43274997, 0.64434578), (0.2201862, 0.23024299, 0.19410873)) # Normalize the dataset so that it has a mean of 0 and a standard deviation of 1 for each channel (RGB)
])

# Define dataset and dataloader
train_dataset = Dataset(annotations_file= FLAGS.train_file, img_dir= FLAGS.img_dir, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

valid_dataset = Dataset(annotations_file= FLAGS.valid_file, img_dir= FLAGS.img_dir, transform=transform)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# Report split sizes
print('Training set has {} instances.'.format(len(train_dataset))) # Print the number of instances in the training dataset
print('Validation set has {} instances.'.format(len(valid_dataset))) # Print the number of instances in the validation dataset

# Initialize our model and move to device
model = MyModel(num_classes=2).to(device)

# Define loss function and optimizer
loss_fn = nn.CrossEntropyLoss() # Define the cross-entropy loss function for multi-class classification
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum) # The momentum parameter in the SGD optimizer determines how much the optimizer should "remember" the previous updates and take them into account when updating the parameters in the current iteration. 
                                                                                     # stochastic gradient descent (SGD)

# Create log file with timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file_name = f"log__{timestamp}.txt"
log_file = open(log_file_name, "w") 
# write the header
log_file.write("Epoch,Train_Loss,Train_Accuracy,Valid_Loss,Valid_Accuracy\n")

# Initialize lists to store statistics
train_losses = []
train_accuracies = []
valid_losses = []
valid_accuracies = []

# Train and validate the model
for epoch in range(num_epochs):
    model.train() # dropout and batch normalization is working
    train_epoch_loss = 0.0
    train_correct = 0
    train_total = 0

    for i, (train_images, train_labels) in enumerate(train_dataloader): # iterationların döndüğü for loop toplam / batchsize
        # Move images and labels to device
        # train_images.shape -> ([16, 3, 224, 224])
        # train_labels.shape -> ([16])
        # print(train_images.shape) 
        # print(train_labels.shape)
        train_images = train_images.to(device)
        train_labels = train_labels.to(device)

        # Forward pass 1
        # train_outputs.shape -> ([16, 2])
        train_outputs = model(train_images)
        # print(train_outputs.shape)

        # Accumulate training loss 2
        # train_epoch_loss.shape -> ('float' object has no attribute 'shape')
        train_loss = loss_fn(train_outputs, train_labels)
        # train_loss.shape -> ([]) 
        # print(train_loss.shape)
        train_epoch_loss += train_loss.item()
        
        # Zero out gradients 3
        optimizer.zero_grad()
        
        # Backward pass 4
        train_loss.backward()
        
        # update weights 5
        optimizer.step() 

        # Update statistics
        # train_predicted.shape -> ([16])
        # train_total.shape -> ('int' object has no attribute 'shape')
        # train_correct.shape -> ('int' object has no attribute 'shape')   
        _, train_predicted = train_outputs.max(1)
        train_total += train_labels.size(0)
        train_correct += train_predicted.eq(train_labels).sum().item()
        # print(train_predicted.shape) 

    # Calculate training accuracy and loss
    train_accuracy = 100 * train_correct / train_total
    train_losses.append(train_epoch_loss / len(train_dataloader))
    train_accuracies.append(train_accuracy)

    model.eval() # dropout and batch normalization is freezed
    valid_epoch_loss = 0.0
    valid_correct = 0
    valid_total = 0

    with torch.no_grad():
        for i, (valid_images, valid_labels) in enumerate(valid_dataloader):
            # Move validation images and labels to device
            valid_images = valid_images.to(device)
            valid_labels = valid_labels.to(device)

            # Forward pass for validation
            valid_outputs = model(valid_images)
            valid_loss = loss_fn(valid_outputs, valid_labels)

            # Accumulate validation loss
            valid_epoch_loss += valid_loss.item()

            # Update statistics
            _, valid_predicted = valid_outputs.max(1)
            valid_total += valid_labels.size(0)
            valid_correct += valid_predicted.eq(valid_labels).sum().item()

        # Calculate validation accuracy and loss
        valid_accuracy = 100 * valid_correct / valid_total
        valid_losses.append(valid_epoch_loss / len(valid_dataloader))
        valid_accuracies.append(valid_accuracy)

    # Print statistics at the end of epoch
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Train Accuracy: {train_accuracies[-1]:.2f}%, Valid Loss: {valid_losses[-1]:.4f}, Valid Accuracy: {valid_accuracies[-1]:.2f}%')
    # Write statistics to log file
    log_file.write(f'{epoch+1},{train_losses[-1]:.4f},{train_accuracies[-1]:.2f},{valid_losses[-1]:.4f},{valid_accuracies[-1]:.2f}\n')

    # Save weights every 10 epochs
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), f"weights__{timestamp}_epoch_{epoch+1}.pt")

    # Save weights when validation loss is good
    if valid_losses[-1] < best_valid_loss:  
        best_valid_loss = valid_losses[-1]
        torch.save(model.state_dict(), f"best_weights__{timestamp}_{best_valid_loss:.4f}.pt")
    
# Close log file
log_file.close()
############ ".item()" can only be called on a tensor with a single element. If you try to call it on a tensor with more than one element, you will get a value error.##################

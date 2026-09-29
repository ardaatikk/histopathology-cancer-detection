import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
import datetime
from pathlib import Path

from model import CancerDetectionModel
from dataset_inference import Dataset

parser = argparse.ArgumentParser(description='')
parser.add_argument(
    '--img_dir',
    default='./images',
    help='Images directory'
)

parser.add_argument(
    '--test_file',
    default='./data/test.csv',
    help='Test annotations file'
)

parser.add_argument(
    '--batch_size',
    type=int,
    default=16,
    help='Batch size'
)

parser.add_argument(
    '--model_file',
    default='./checkpoints/weights_epoch_100.pt',
    help='Model checkpoint'
)

parser.add_argument(
    '--output_dir',
    default='./outputs',
    help='Directory for prediction outputs'
)

FLAGS = parser.parse_args()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define hyperparameters
batch_size = FLAGS.batch_size

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize((0.62376275, 0.43274997, 0.64434578), (0.2201862, 0.23024299, 0.19410873))
])

# Define dataset and dataloader
test_dataset = Dataset(annotations_file= FLAGS.test_file, img_dir= FLAGS.img_dir, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model and load the saved weights
model = CancerDetectionModel(num_classes=2).to(device)
model.load_state_dict(torch.load(FLAGS.model_file,  map_location=torch.device('cpu')))  

# Initialize lists to store the test results
test_results = []

# Create log file with timestamp
# /Users/ardaatik/Desktop/my_projects/classification/weights__20230802_133002_epoch_100.pt
model_id = FLAGS.model_file.split('__')[-1][:-3]
model_name = Path(FLAGS.model_file).stem
output_dir = Path(FLAGS.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
log_file_name = output_dir / f"predictions__{model_name}.csv"
log_file = open(log_file_name, "w") 
# write the header
log_file.write("img_id,cancer_score\n")

# Evaluation mode
model.eval()
with torch.no_grad():
    for i, (images, img_ids) in enumerate(test_dataloader):
        # Move validation images and labels to device
        images = images.to(device)

        # Forward pass 
        test_outputs = model(images)
        normalized_output = F.softmax(test_outputs, 1)
        
        for j in range(len(img_ids)):
            temp_img_id = img_ids[j]
            temp_score = normalized_output[j]        
            log_file.write(f'{temp_img_id},{temp_score[1]}\n')


# Close the log file
log_file.close()

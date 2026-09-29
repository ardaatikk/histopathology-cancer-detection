import pandas as pd
import torch
import os

from torch.utils.data import Dataset
from torchvision.io import read_image


class Dataset(torch.utils.data.Dataset): 
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        super().__init__()
        self.img_dir = img_dir #image directory
        self.transform = transform #transforms applied to the images
        self.img_labels = pd.read_csv(annotations_file) #getting labels from train.csv file
        self.target_transform = target_transform 
        
    def __len__(self):
        return len(self.img_labels) #getting the length
    
    def __getitem__(self, idx): #idx is the index of a specific image and its label
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0] + '.jpeg')
        #image = read_image(img_path).type(torch.FloatTensor)/255. # reading an image as RGB
        image = read_image(img_path)/255. # reading an image as RGB
        label = self.img_labels.iloc[idx, 1] # getting the value of first column
        if self.transform:
            image = self.transform(image)
        # if self.target_transform:
        #     label = self.target_transform(label)
        return image, label
        # if parts applies the transformations if there are any


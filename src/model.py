import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, num_classes=2, num_nodes=128):
        super(Model, self).__init__()
    
        self.resnet18 = models.resnet18(pretrained=True)
        self.resnet18 = nn.Sequential(*list(self.resnet18.children())[:-2])
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc1 = nn.Linear(512 * 1 * 1 , num_nodes)  # Calculate the input size based on ResNet-18 output shape
        self.fc2 = nn.Linear(num_nodes, num_classes)
    def forward(self, x):
        x = self.resnet18(x)
        x = self.avgpool(x) 
        x = torch.flatten(x, 1)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        return x

class MyModel(nn.Module):
    def __init__(self, num_classes=2, num_nodes=128):
        super(MyModel, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, 3, 1) # 3 input channels as RGB 32 output 3 kernel 1 stride (stride is 1 default)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
      
        self.maxpool1 = nn.MaxPool2d(2, 2) # Applies a 2D max pooling over an input signal composed of several input planes.
        self.maxpool2 = nn.MaxPool2d(2, 2) 
        
        self.fc1 = nn.Linear(186624, num_nodes)  # 186624   1016064
        self.fc2 = nn.Linear(num_nodes, num_classes)
        
    def forward(self, x):  # first apply the convolutional layers then pooling layers. After that flatten the output before passing through linear layers.
        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool2(x)

        x = torch.flatten(x, 1) # flatten is important because we convert the output to 1 dimension so our fc1 and fc2 can work
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        # output = F.log_softmax(x, dim=1) # Specifically, taking the logarithm of the softmax values allows for easier computation of gradients during the training process.
        return x                   # returns a Tensor of the same dimension and shape as the input with values in the range [-inf, 0)

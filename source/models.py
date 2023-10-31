import torch
from torch import nn 


class CBM(nn.Module):
    def __init__(self, c_chanel: int, 
                 hidden_units: int, 
                 concepts:int, 
                 output_shape: int,
                 img_shape: int):
        """
        c_chane (int): input chanel size
        hidden_units (int): hidden units in CNN layers
        concepts (int): total number of concepts
        output_shape (int): total number of target classes
        img_shape (int): transform images to a specific dimension   
        """
        super().__init__()
        self.conv1= nn.Conv2d(in_channels=c_chanel, 
                              out_channels=hidden_units, 
                              kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=hidden_units, 
                               out_channels=64, kernel_size=3)
        self.dropout1 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(in_features=64 * (int) ((img_shape-2)/2) * (int) ((img_shape-2)/2), 
                             out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=concepts)
        self.fc3 = nn.Linear(in_features=concepts, out_features=output_shape)
        
    def forward(self, x):
        x= self.conv1(x)
        x= nn.functional.relu(x)
        x=self.conv2(x)
        x= nn.functional.relu(x)
        x = nn.functional.max_pool2d(x,2)
        x= self.dropout1(x)
        x= torch.flatten(x, 1)
        x= self.fc1(x)
        x= nn.functional.relu(x)
        x= self.fc2(x)
        c_hat = nn.functional.relu(x)
        x = self.fc3(c_hat)
        return x, c_hat

    # def forward_until_concepts(self, x):
    #     x= self.conv1(x)
    #     x= nn.functional.relu(x)
    #     x=self.conv2(x)
    #     x= nn.functional.relu(x)
    #     x = nn.functional.max_pool2d(x,2)
    #     x= self.dropout1(x)
    #     x= torch.flatten(x, 1)
    #     x= self.fc1(x)
    #     x= nn.functional.relu(x)
    #     x= self.fc2(x)
    #     x = nn.functional.relu(x)
    #     return x
    
        
class CBM_for_AL(nn.Module):
    def __init__(self, c_chanel=3, 
                 hidden_units=32, 
                 concepts=312, 
                 output_shape=200,
                 img_shape=32):
        """
        c_chane (int): input chanel size
        hidden_units (int): hidden units in CNN layers
        concepts (int): total number of concepts
        output_shape (int): total number of target classes
        img_shape (int): transform images to a specific dimension   
        """
        super().__init__()
        self.conv1= nn.Conv2d(in_channels=c_chanel, 
                              out_channels=hidden_units, 
                              kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=hidden_units, 
                               out_channels=64, kernel_size=3)
        self.dropout1 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(in_features=64 * (int) ((img_shape-2)/2) * (int) ((img_shape-2)/2), 
                             out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=concepts)
        self.fc3 = nn.Linear(in_features=concepts, out_features=output_shape)
        
    def forward(self, x):
        x= self.conv1(x)
        x= nn.functional.relu(x)
        x=self.conv2(x)
        x= nn.functional.relu(x)
        x = nn.functional.max_pool2d(x,2)
        x= self.dropout1(x)
        x= torch.flatten(x, 1)
        x= self.fc1(x)
        x= nn.functional.relu(x)
        x= self.fc2(x)
        c_hat = nn.functional.relu(x)
        x = self.fc3(c_hat)
        return x
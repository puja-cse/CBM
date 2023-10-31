from torchvision import datasets
from torchvision.transforms import ToTensor
import torch

def GET_MNIST():
    def __init__(self, root='../dataset/MNIST'):
        train_data = datasets.MNIST( root=root,
                                    train="True",
                                    transform=ToTensor,
                                    download=True)
        test_data = datasets.MNIST(root=root,
                                   train=False,
                                   transform=ToTensor)     
def get_dataloaders(self):
    loaders = {'train': torch.utils.data.DataLoader(
                self.train_data, 
                batch_size =100,
                shuffle=True,
                num_workers=1),
                
                'test' : torch.utils.data.DataLoader(self.test_data,
                                                        batch_size=100,
                                                        shuffle=True,
                                                        num_workers=1)
    }
    return loaders
    

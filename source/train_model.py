#training loop

from tqdm.auto import tqdm
from timeit import default_timer as timer
import torch
import random
from torch import nn
from torchvision import transforms
from PIL import Image
import numpy
import matplotlib.pyplot as plt  
from source import data_processing

def train_with_MNIST(epochs, model, train_dataloader, device): #for MNIST
    torch.manual_seed=42
    torch.cuda.manual_seed=42
    # train_time_start_on_cpu = timer()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)

    #epochs =1 
    for epoch in tqdm(range(epochs)):
        print(f"Epoch: {epoch}\n-----")
        train_loss=0
        for batch, (X,y) in enumerate(train_dataloader):
            model.train()
            X, y = X.to(device), y.to(device)
            y_logits = model(X)
            #print(f"Logit shape: {y_logits.shape}, Y shape: {y.shape}")
            loss = loss_fn(y_logits, y)
            train_loss+=loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch%500==0:
                print(f"Looked at {batch*len(X)}/{len(train_dataloader.dataset)} samples\n")
        train_loss /= len(train_dataloader)


def train_with_CUB(model,
                   X_train,
                   y_train,
                   epochs =10,
                   BATCH_SIZE = 32,
                   device = 'cpu',
                   img_shape=28,
                   learning_rate = 0.01,
                   consider_concept_loss = True, 
                   consider_target_loss = True,
                   c_lambda = 1.0):
    """
        model (cbm): an object of CBM class
        epochs (int): total number of epochs
        BATCH_SIZE (int): number of images per batch
        device (str): 'cuda' or 'cpu' (default cpu)
        img_shape (int): what should be reduced size of the image (default 28x28)
        consider_concept_loss : whether to compute the concept loss or not  
    """
    torch.manual_seed=42
    torch.cuda.manual_seed=42
    
    
    loss_fn_concept = nn.BCEWithLogitsLoss()  #-------- loss function for concept identification
    loss_fn_target = nn.CrossEntropyLoss()    #-------- loss function for class label identifaction
    
    
    optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate) #------ optimizer SGD (Stochhastic Gradient Descent)

    total_train_data =len(X_train)
    total_batch = int (total_train_data/BATCH_SIZE)

    resize = transforms.Resize( size=(img_shape, img_shape) ) #---- resize the image 
    #gray = transforms.Grayscale() #----- convert into gray scale 
    convert_tensor = transforms.ToTensor() #------- transform the image into tensor

    loss_values=[]
    concept_loss = []
    target_loss = []

    #-------------LOAD THE DATASET (FULL TRAIN DATASET) FROM THE GIVEN PATH -----------
    X_list = []
    concept_list = []
    target_list = [] 
    for i in range (len(X_train)):
        img = Image.open(X_train[i]['img_path'])
        img = convert_tensor(resize(img))
        img_np = img.numpy()
        X_list.append(img_np)
        concept_list.append(X_train[i]['attribute_label'])
        target_list.append(y_train[i])
    
    X_train = torch.tensor(X_list).to(device)
    concepts = torch.tensor(concept_list).to(device)
    y_train = torch.tensor(target_list).to(device)

    del X_list, concept_list, target_list

    #---------------TRAINING LOOP ------------###
    
    for epoch in range(epochs):
        indices = torch.randperm(X_train.size()[0])  #----- shuffling the indices number randomly------
        X_train = X_train[indices]
        concepts = concepts[indices]
        y_train = y_train[indices]
        train_loss=0.0
        concept_loss_per_epoch =0.0
        target_loss_per_epoch = 0.0
        #print(f"------Epoch {epoch+1} is in progress--------")
        for batch in range(total_batch):
            X= X_train[ batch*BATCH_SIZE : batch*BATCH_SIZE+BATCH_SIZE ]
            c = concepts[ batch*BATCH_SIZE : batch*BATCH_SIZE+BATCH_SIZE ]
            y = y_train[ batch*BATCH_SIZE : batch*BATCH_SIZE+BATCH_SIZE ]

            model.train()  #----- putting model into training mode ------
            y_logits, c_hats = model(X)
            
            optimizer.zero_grad()

            loss1 = loss_fn_concept(c_hats, c.float())
            loss2 = loss_fn_target(y_logits, y)
            if consider_concept_loss and consider_target_loss: 
                loss1 = c_lambda*loss1
                loss = loss1 + loss2
            elif consider_concept_loss:
                loss = loss1
            else:
                loss = loss2
            concept_loss_per_epoch +=loss1
            target_loss_per_epoch += loss2
            train_loss += loss
            loss.backward()

            optimizer.step()
            # if batch%50==0:
            #     print(f"Total {(batch+1)*BATCH_SIZE} of {total_batch*BATCH_SIZE} data processed")
        
        train_loss /= total_batch   #--------- train_loss /= (total_batch*BATCH_SIZE)
        concept_loss_per_epoch /= total_batch #------  concept_loss_per_epoch /= (total_batch*BATCH_SIZE)
        target_loss_per_epoch /= total_batch  #------- target_loss_per_epoch /= (total_batch*BATCH_SIZE)
        loss_values.append(train_loss)
        concept_loss.append(concept_loss_per_epoch)
        target_loss.append(target_loss_per_epoch)

        print(f"Epoch {epoch+1}: Training loss {train_loss: .5f}, concept loss: {concept_loss_per_epoch: .5f}, target loss: {target_loss_per_epoch: .5f}\n")
    print("------Training Finished-----")
    del X_train, y_train, c 
    return loss_values, concept_loss, target_loss
    
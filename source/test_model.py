
import torch
from torch import nn 
from torchvision import transforms
from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
from source.data_processing import extract_attribute_label_ranges


#########################################
##########-----EDIT FROM HERE-----################
#####################################

def test_with_CUB(model,
                  X_test,
                  y_test,
                  output_dir, 
                  device='cpu',
                  n_concepts=312,
                  n_labels=200,
                  img_shape=32,
                  c_lambda =1.0):
    
    loss_fn_concept = nn.BCEWithLogitsLoss()  #-------- loss function for concept identification
    loss_fn_target = nn.CrossEntropyLoss()    #-------- loss function for class label identifaction
    

    resize = transforms.Resize( size=(img_shape,img_shape) ) #---- resize the image 
    #gray = transforms.Grayscale() #----- convert into gray scale 
    convert_tensor = transforms.ToTensor() #------- transform the image into tensor
    
    X_list =[]
    target_list =[]
    concept_list =[]
    for i in range (len(X_test)):
        img = Image.open(X_test[i]['img_path'])
        img = convert_tensor(resize(img))
        img_np = img.numpy()
        X_list.append(img_np)
        concept_list.append(X_test[i]['attribute_label'])
        target_list.append(y_test[i])
    X_test = torch.tensor(X_list).to(device)
    c = torch.tensor(concept_list).to(device)
    y_true = torch.tensor(target_list).to(device)
    del X_list, target_list, concept_list
    model = model.to(device)
    model.eval()
    with torch.inference_mode():
        y_logits, c_hats = model(X_test)
        loss1 = c_lambda*loss_fn_concept(c_hats, c.float())
        loss2 = loss_fn_target(y_logits, y_true)

        y_pred_array = y_logits.argmax(dim=1).detach().cpu().numpy()
        y_true_array = y_true.detach().cpu().numpy()
        labels = np.arange(n_labels)
        CF = confusion_matrix(y_true=y_true_array, y_pred=y_pred_array, labels=labels)
        test_acc = accuracy_score(y_true=y_true_array, y_pred=y_pred_array)
        precisions, recalls, f1_score, support = precision_recall_fscore_support(y_true=y_true_array, y_pred=y_pred_array)

        file = open(output_dir, "a")

        file.write(f"TEST LOSS : {loss1+loss2:.5f} CONCEPT LOSS: {loss1: .5f} TARGET LOSS: {loss2: .5f} where c_lambda: {c_lambda:0.2}\n")
        file.write(f"------------PERFORMANCE MATRICES FOR TARGETS ----------\n")        
        file.write(f"TARGET ACCURACY: {test_acc: .5f}")
        file.write(f"\nPrecisions:\n{precisions}")
        file.write(f"\nRecalls:\n{recalls}")
        file.write(f"\nF1- Score: {f1_score}")
        file.write(f"\n------CONFUSION MATRIX for TARGET -------\n{CF}")

        ###------- CONFUSION MATRIX FOR EACH CONCEPT/ATTRIBUTE ----------####
        CONCEPT_GROUP_MAP, idx_to_attribute, attribute_to_idx = extract_attribute_label_ranges()
        for concept in CONCEPT_GROUP_MAP:
            low_range = min(CONCEPT_GROUP_MAP[concept])
            high_range = max(CONCEPT_GROUP_MAP[concept])
            indices = torch.arange(low_range-1, high_range, 1)
            predicted_portion = torch.index_select(c_hats, dim=1, index=indices)
            true_portion = torch.index_select(c, dim=1, index=indices)
            predicted_concepts = predicted_portion.argmax(dim=1).detach().cpu().numpy()
            true_concepts = true_portion.argmax(dim=1).detach().cpu().numpy()
            pseudo_labels = np.arange(len(CONCEPT_GROUP_MAP[concept]))
            file.write(f"\n-------PERFORMANCE MATRICES FOR: {concept} ----------\n")
            file.write("The attributes are: \n")
            for idx in CONCEPT_GROUP_MAP[concept]:
                file.write(idx_to_attribute[idx])
            CF = confusion_matrix(y_true=true_concepts, y_pred=predicted_concepts, labels=pseudo_labels)
            file.write(f"\nConcept accuracy: {accuracy_score(y_true=true_concepts, y_pred=predicted_concepts)}")
            precisions, recalls, f1_score, support = precision_recall_fscore_support(y_true=true_concepts, y_pred=predicted_concepts)
            file.write(f"\nPrecision: {precisions}")
            file.write(f"\nRecalls:\n{recalls}")
            file.write(f"\nF1- Score:\n{f1_score}")
            file.write(f"------CONFUSION MATRIX for {concept} -------\n{CF}")
            file.write("\n**********************************************\n")
        file.close()
    del model, X_test, y_test, c, precisions, recalls, f1_score
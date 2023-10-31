"""
Make train, val, test datasets based on train_test_split.txt, and by sampling
val_ratio of the official train data to make a validation set.
Each dataset is a list of metadata, each includes official image id, full image
path, class label, attribute labels, attribute certainty scores, and attribute
labels calibrated for uncertainty
Taken from: https://github.com/yewsiang/ConceptBottleneck/blob/master/CUB/data_processing.py
"""
import os
import random
from os import listdir
from os.path import isfile, isdir, join
from collections import defaultdict as ddict
import torch
from PIL import Image
from sklearn.model_selection import train_test_split

def extract_data(data_dir):
    cwd = os.getcwd()
    data_path = join(cwd, data_dir + '/images')
    val_ratio = 0.2

    path_to_id_map = dict() #-----map from full image path to image id
    with open(data_path.replace('images', 'images.txt'), 'r') as f:
        for line in f:
            items = line.strip().split()
            key_str = join(data_path, items[1]).replace('\\', '/')
            path_to_id_map[key_str] = int(items[0])

    attribute_labels_all = ddict(list)      #---------------------map from image id to a list of attribute labels
    attribute_certainties_all = ddict(list)   #-------------------map from image id to a list of attribute certainties
    attribute_uncertain_labels_all = ddict(list) #----------------map from image id to a list of attribute labels calibrated for uncertainty
    # 1 = not visible, 2 = guessing, 3 = probably, 4 = definitely
    uncertainty_map = {1: {1: 0, 2: 0.5, 3: 0.75, 4:1}, #calibrate main label based on uncertainty label
                        0: {1: 0, 2: 0.5, 3: 0.25, 4: 0}}
    with open(join(cwd, data_dir + '/attributes/image_attribute_labels.txt'), 'r') as f:
        for line in f:
            file_idx, attribute_idx, attribute_label, attribute_certainty = line.strip().split()[:4]
            attribute_label = int(attribute_label)
            attribute_certainty = int(attribute_certainty)
            uncertain_label = uncertainty_map[attribute_label][attribute_certainty]
            attribute_labels_all[int(file_idx)].append(attribute_label)
            attribute_uncertain_labels_all[int(file_idx)].append(uncertain_label)
            attribute_certainties_all[int(file_idx)].append(attribute_certainty)

    is_train_test = dict() #map from image id to 0 / 1 (1 = train)
    with open(join(cwd, data_dir + '/train_test_split.txt'), 'r') as f:
        for line in f:
            idx, is_train = line.strip().split()
            is_train_test[int(idx)] = int(is_train)
    print("Number of train images from official train test split:", sum(list(is_train_test.values())))

    train_val_data, test_data = [], []
    train_data, val_data = [], []
    folder_list = [f for f in listdir(data_path) if isdir(join(data_path, f))]
    folder_list.sort() #sort by class index
    for i, folder in enumerate(folder_list):
        folder_path = join(data_path, folder)
        classfile_list = [cf for cf in listdir(folder_path) if (isfile(join(folder_path,cf)) and cf[0] != '.')]
        #classfile_list.sort()
        for cf in classfile_list:
            key_str = join(folder_path, cf).replace('\\', '/')
            img_id = path_to_id_map[key_str]
            img_path = join(folder_path, cf).replace('\\', '/')
            img = Image.open('img_path')
            if img.mode != 'RGB':  #------- since there are a few gray scale images, we just skipped them
                continue
            metadata = {'id': img_id, 'img_path': img_path, 'class_label': i,
                      'attribute_label': attribute_labels_all[img_id], 'attribute_certainty': attribute_certainties_all[img_id],
                      'uncertain_attribute_label': attribute_uncertain_labels_all[img_id]}
            if is_train_test[img_id]:
                train_val_data.append(metadata)
                # if val_files is not None:
                #     if img_path in val_files:
                #         val_data.append(metadata)
                #     else:
                #         train_data.append(metadata)
            else:
                test_data.append(metadata)

    random.shuffle(train_val_data)
    return train_val_data, test_data
    # split = int(val_ratio * len(train_val_data))
    # train_data = train_val_data[split :]
    # val_data = train_val_data[: split]
    # print('Size of train set:', len(train_data))
    # return train_data, val_data, test_data

def extract_attribute_label_ranges(data_dir="../dataset/cub_200_2011"):
    """
    this function loads the text file containing the list of attributes/ concepts and returns 3 dictionaries,
    namely:
        Concept_Group_Map: Which is a dictionary where the key value is the name of the unique concept i.e. 
                            Concept_Group_Map['has_wing_pattern'] = [309, 310, 311, 312]
        idx_to_attribute: Maps idx to attribute/concept name 
                          i.e. idx_to_attribute['has_wing_pattern::solid']=309
        attribute_to_idx: Maps attribute/concept name to idx
                          i.e.  idx_to_attribute[309]='has_wing_pattern::solid'
    """
    idx_to_attribute = dict()
    attribute_to_idx = dict()
    CONCEPT_GROUP_MAP = ddict(list)
    CONCEPT_GROUP_MAP 
    cwd = os.getcwd()
    with open(join(cwd, data_dir + '/attributes/attributes.txt'), 'r') as f:
        for line in f:
            attribute_idx, concept_with_label = line.strip().split(' ')
            concept = concept_with_label.split("::")[0]
            attribute_idx= int(attribute_idx)
            idx_to_attribute[attribute_idx] = concept_with_label
            attribute_to_idx[concept_with_label] = attribute_idx
            CONCEPT_GROUP_MAP[concept].append(attribute_idx)
    return CONCEPT_GROUP_MAP, idx_to_attribute, attribute_to_idx


def get_concept_names(c_hat ,
                      true_concepts ,
                      CONCEPT_GROUP_MAP ,
                      idx_to_attribute ):
    """
        Given c_hat and true_concept list for a single data/image, this function is intended to
        return the list of predicted concepts and actual concept
    """
    predicted_concept_list=[]
    actual_concept_list=[]
    for keys in CONCEPT_GROUP_MAP:
        low_range = min(CONCEPT_GROUP_MAP[keys])
        high_range = max(CONCEPT_GROUP_MAP[keys])
        predicted = torch.argmax(c_hat[low_range-1 : high_range])
        concept = torch.argmax(true_concepts[low_range-1, high_range])
        predicted_concept_list.append( idx_to_attribute[ CONCEPT_GROUP_MAP[keys][predicted] ])
        actual_concept_list.append( idx_to_attribute[ CONCEPT_GROUP_MAP[keys][concept] ])
    return predicted_concept_list, actual_concept_list



def extract_data_path_concept_and_label(data_dir,
                                    test_ratio = 0.2,
                                    stratify = True):
    cwd = os.getcwd()
    data_path = join(cwd, data_dir + '/images')
    val_ratio = 0.2

    path_to_id_map = dict() #-----map from full image path to image id
    with open(data_path.replace('images', 'images.txt'), 'r') as f:
        for line in f:
            items = line.strip().split()
            key_str = join(data_path, items[1]).replace('\\', '/')
            path_to_id_map[key_str] = int(items[0])

    attribute_labels_all = ddict(list)      #---------------------map from image id to a list of attribute labels
    attribute_certainties_all = ddict(list)   #-------------------map from image id to a list of attribute certainties
    attribute_uncertain_labels_all = ddict(list) #----------------map from image id to a list of attribute labels calibrated for uncertainty
    # 1 = not visible, 2 = guessing, 3 = probably, 4 = definitely
    uncertainty_map = {1: {1: 0, 2: 0.5, 3: 0.75, 4:1}, #calibrate main label based on uncertainty label
                        0: {1: 0, 2: 0.5, 3: 0.25, 4: 0}}
    with open(join(cwd, data_dir + '/attributes/image_attribute_labels.txt'), 'r') as f:
        for line in f:
            file_idx, attribute_idx, attribute_label, attribute_certainty = line.strip().split()[:4]
            attribute_label = int(attribute_label)
            attribute_certainty = int(attribute_certainty)
            uncertain_label = uncertainty_map[attribute_label][attribute_certainty]
            attribute_labels_all[int(file_idx)].append(attribute_label)
            attribute_uncertain_labels_all[int(file_idx)].append(uncertain_label)
            attribute_certainties_all[int(file_idx)].append(attribute_certainty)

    is_train_test = dict() #map from image id to 0 / 1 (1 = train)
    with open(join(cwd, data_dir + '/train_test_split.txt'), 'r') as f:
        for line in f:
            idx, is_train = line.strip().split()
            is_train_test[int(idx)] = int(is_train)
    print("Number of train images from official train test split:", sum(list(is_train_test.values())))

    X_metadata=[]
    y_labels =[]
    folder_list = [f for f in listdir(data_path) if isdir(join(data_path, f))]
    folder_list.sort() #sort by class index
    for i, folder in enumerate(folder_list):
        folder_path = join(data_path, folder)
        classfile_list = [cf for cf in listdir(folder_path) if (isfile(join(folder_path,cf)) and cf[0] != '.')]
        #classfile_list.sort()
        for cf in classfile_list:
            key_str = join(folder_path, cf).replace('\\', '/')
            img_id = path_to_id_map[key_str]
            img_path = join(folder_path, cf).replace('\\', '/')
            img = Image.open(img_path)
            if img.mode != 'RGB':  #------- since there are a few gray scale images, we just skipped them
                continue
            metadata = {'id': img_id, 'img_path': img_path,
                      'attribute_label': attribute_labels_all[img_id] }
            X_metadata.append(metadata)
            y_labels.append(i)
    X_train, X_test, y_train, y_test = train_test_split(X_metadata, y_labels, train_size=0.8, random_state=42, stratify=y_labels)
    return X_train, X_test, y_train, y_test 
import os,glob
import random
import pandas as pd

annotaion_path = 'C:/PROJECT_CODE/DETECTION_NET/Helen-Images/annotation'
training_paths = ['C:/PROJECT_CODE/DETECTION_NET/Helen-Images/test_train']
validation_paths = ['C:/PROJECT_CODE/DETECTION_NET/Helen-Images/test_val']
# training_paths = ['C:/PROJECT_CODE/DETECTION_NET/Helen-Images/train_1']
# validation_paths = ['C:/PROJECT_CODE/DETECTION_NET/Helen-Images/test']

def name_encoder(name_master_list):
    # Flatten train_name_master_list
    flattened_names = [name for sublist in name_master_list for name in sublist]
    # Generate a mapping from original names to encoded integers
    name_to_encoded = {name: i for i, name in enumerate(sorted(set(flattened_names)))}
    # Encode the list of names while maintaining the structure of train_name_master_list
    encoded_names = [[name_to_encoded[name] for name in sublist] for sublist in name_master_list]
    # Generate a list of unique encoded names
    unique_names = list(range(len(set(flattened_names))))
    return unique_names,encoded_names

annotations = glob.glob(annotaion_path+'/*.txt')
annotations[0]
annotations_master_file = []
for file in annotations:
    pd_file = pd.read_csv(file)
    image_name = pd_file.columns[0]
    xloc = pd_file.index.to_list()
    yloc = pd_file.iloc[:, 0].tolist()
    annotations_master_file.append([image_name,xloc,yloc])


train_image_master_list = []
train_name_master_list = []
for path in training_paths:
    train_images = glob.glob(path+'/*.jpg')
    train_image_path=[]
    train_names=[]
    for image in train_images:
        train_image_path.append(image.replace('\\','/'))
        train_names.append(os.path.basename(image))
    train_image_master_list.append(train_image_path)
    train_name_master_list.append(train_names)

val_image_master_list = []
val_name_master_list = []
for path in validation_paths:
    val_images = glob.glob(path+'/*.jpg')
    val_image_path=[]
    val_names=[]
    for image in val_images:
        val_image_path.append(image.replace('\\','/'))
        val_names.append(os.path.basename(image))
    val_image_master_list.append(val_image_path)
    val_name_master_list.append(val_names)

training_unique_names,training_encoded_names = name_encoder(train_name_master_list)
val_unique_names,val_encoded_names = name_encoder(val_name_master_list)



train_matching_indices = []
val_matching_indices = []
for i in range(len(annotations_master_file)):
    for j in range(len(train_name_master_list)):
        for k in range(len(train_name_master_list[0])):
            if train_name_master_list[j][k].find(annotations_master_file[i][0]) != -1:
                train_matching_indices.append([i,j,k])
    for j in range(len(val_name_master_list)):
        for k in range(len(val_name_master_list[0])):
            if val_name_master_list[j][k].find(annotations_master_file[i][0]) != -1:
                val_matching_indices.append([i,j,k])


train_ordered_path = []
train_ordered_annotation = []
train_label = []
for i in range(len(train_matching_indices)):
    # print(i)
    temp = [train_image_master_list[train_matching_indices[i][1]][train_matching_indices[i][2]]]
    temp_xy = annotations_master_file[train_matching_indices[i][0]][1],annotations_master_file[train_matching_indices[i][0]][2]
    label=0
    train_ordered_path.append(temp)
    train_ordered_annotation.append(temp_xy)
    train_label.append(label)
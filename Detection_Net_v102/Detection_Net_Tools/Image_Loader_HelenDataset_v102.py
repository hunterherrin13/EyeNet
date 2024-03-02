import os,glob
import random
import pandas as pd

annotaion_path = 'C:/PROJECT_CODE/DETECTION_NET/Helen-Images/annotation'
training_paths = ['C:/PROJECT_CODE/DETECTION_NET/Helen-Images/train_1']

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
    train_image_master_list.append(train_images)
    train_names=[]
    for image in train_images:
        train_names.append(os.path.basename(image))
    train_name_master_list.append(train_names)

print(train_name_master_list)


matching_indices = []

for train_names in train_name_master_list:
    for name in train_names:
        for i, (image_name, _, _) in enumerate(annotations_master_file):
            if name in image_name:
                matching_indices.append(i)
                break  # Move to the next train_names group

print(matching_indices)
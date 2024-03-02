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

# print(train_name_master_list)


matching_indices = []
for i in range(len(train_name_master_list)):
    for j in range(len(annotations_master_file)):
        for k in range(len(train_name_master_list[0])):
            if train_name_master_list[i][k].find(annotations_master_file[j][0]) != -1:
                # print(j)
                # print(k)
                matching_indices.append([i,j,k])

# print("Matching indices:", matching_indices)
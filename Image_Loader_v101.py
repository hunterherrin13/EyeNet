import os,glob

path = 'C:/PROJECT_CODE/DETECTION_NET/FACES_20240224'

train_images = glob.glob(path+'/*.jpg')

names=[]
for image in train_images:
    name = os.path.basename(image).split('/')[0].split('_')[0]
    names.append(int(name))

num_names = len(set(names))
unique_names = list(range(len(set(names))))
print(unique_names)
# print(num_names)
# print(set(names))
# print(images)
# print(names)
val_images = [train_images[0],train_images[12],train_images[24],train_images[36],train_images[48]]
val_names = unique_names

# print(val_images)
# print(val_names)
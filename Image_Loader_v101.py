import os,glob

path = 'C:/PROJECT_CODE/DETECTION_NET/FACES_20240224'

train_images = glob.glob(path+'/*.jpg')

names=[]
for image in train_images:
    name = os.path.basename(image).split('/')[0].split('_')[0]
    names.append(int(name))

# Generate a mapping from original names to encoded integers
name_to_encoded = {name: i for i, name in enumerate(sorted(set(names)))}

# Encode the list of names
encoded_names = [name_to_encoded[name] for name in names]
unique_names = list(range(len(set(names))))

val_images = [train_images[0],train_images[12],train_images[24],train_images[36],train_images[48],train_images[60]]
val_names = unique_names

# print(val_images)
# print(val_names)
import os,glob
import random

path = 'C:/PROJECT_CODE/DETECTION_NET/FACES_20240224'

train_images = glob.glob(path+'/*.jpg')
train_images = train_images[0:23]

names=[]
for image in train_images[0:23]:
    name = os.path.basename(image).split('/')[0].split('_')[0]
    names.append(int(name))

# Generate a mapping from original names to encoded integers
name_to_encoded = {name: i for i, name in enumerate(sorted(set(names)))}

# Encode the list of names
encoded_names = [name_to_encoded[name] for name in names]
unique_names = list(range(len(set(names))))

val_images = train_images
val_names = encoded_names

combined_data = list(zip(val_images, val_names))

# Shuffle the combined list in place
random.shuffle(combined_data)

# Unzip the shuffled list to get the shuffled val_images and val_names
shuffled_val_images, shuffled_val_names = zip(*combined_data)

# print(encoded_names)
# print(train_images)
# print(val_names)
# print(val_images)
test_path = 'C:/PROJECT_CODE/DETECTION_NET/TestImages/'

test_images = [test_path+'cow1.jpg',test_path+'cow2.jpg',test_path+'cow3.jpg',test_path+'snake1.jpg',test_path+'snake2.jpg',test_path+'snake2.jpg']
test_names = [0,0,0,1,1,1]
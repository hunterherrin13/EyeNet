from torchvision import transforms

def im_transform(resize=(240,240),mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]):
   
   return transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(resize),
    transforms.Normalize(mean, std)
    ])

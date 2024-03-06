import torch
import cv2
import numpy as np
from FaceLandmark_Functions_v102 import FacialLandmarkNet, val_transform


lr=0.01
num_epochs = 20
best_model_path = 'C:/PROJECT_CODE/DETECTION_NET/Models/facelandmark_model.pth'

# Initialize the CNN model
num_classes = 1
num_landmarks = 194

# Load the saved model
checkpoint = torch.load(best_model_path)
model = FacialLandmarkNet(num_classes, num_landmarks)  # Make sure to replace 'num_classes' and 'num_landmarks' with appropriate values
# print(checkpoint['model_state_dict'].keys())
# print(model.state_dict().keys())
model.load_state_dict(checkpoint['model_state_dict'])
epoch = checkpoint['epoch']
optimizer_state_dict = checkpoint['optimizer_state_dict']
best_val_loss = checkpoint['best_val_loss']
train_loss = checkpoint['train_loss']
val_loss = checkpoint['val_loss']
metrics = checkpoint['metrics']
date_time = checkpoint['date_time']
model.eval()

def preprocess_image(image):
    # Resize image to a fixed size if necessary
    # Assuming you want to resize to (224, 224)
    resized_image = cv2.resize(image, (224, 224))
    # Normalize image if necessary (convert to float and scale to [0, 1] range)
    normalized_image = resized_image.astype(np.float32) / 255.0
    # You may need to further preprocess the image depending on your model's requirements
    return normalized_image

def predict_landmarks(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path)
    preprocessed_image = preprocess_image(image)
    # Convert the preprocessed image to a tensor
    tensor_image = torch.tensor(preprocessed_image, dtype=torch.float)
    # Add batch dimension
    tensor_image = tensor_image.unsqueeze(0)
    # Forward pass through the model
    with torch.no_grad():
        outputs = model(tensor_image)
        print(outputs)
    return outputs

# Example usage
image_path = 'path_to_your_image.jpg'
predicted_landmarks = predict_landmarks(image_path)
print(predicted_landmarks)
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


def predict_landmarks(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor_image = val_transform(image)
    # Add batch dimension
    tensor_image = tensor_image.unsqueeze(0)
    # Forward pass through the model
    with torch.no_grad():
        outputs = model(tensor_image)
    # Print shape and contents of the output tensor
    # print("Output tensor shape:", outputs.shape)
    # print("Output tensor contents:", outputs)
    # Extract predicted landmarks from the output tensor
    landmarks_predictions = outputs[:, -num_landmarks*2:]  # Assuming landmarks are concatenated to the end of the output tensor
    # Reshape landmarks predictions
    landmarks_predictions = landmarks_predictions.view(-1, num_landmarks, 2)  # Reshape to match the format of the landmarks
    # Display the image for visualization purposes
    # cv2.imshow("Image", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    print(landmarks_predictions)

    return outputs

# Example usage
image_path = 'C:/PROJECT_CODE/DETECTION_NET/Helen-Images/train_1/10406776_1.jpg'
predicted_landmarks = predict_landmarks(image_path)
print(predicted_landmarks)
import torch
import cv2
import numpy as np
from FaceLandmark_Functions_v102 import FacialLandmarkNet, train_transform

# Function to preprocess input image for inference
def preprocess_image_inference(image_path, transform=None):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (240, 240))  # Resize image to match training size

    if transform:
        image = transform(image)

    return image

# Function to predict landmarks using the trained model
def predict_landmarks(model, image_tensor, num_landmarks):
    dummy_landmarks = torch.zeros(image_tensor.size(0), num_landmarks * 2)  # Dummy landmarks
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor, dummy_landmarks)  # Pass dummy landmarks
    
    # Separate image features and landmark features
    image_features = outputs[:, :-num_landmarks*2]
    landmark_features = outputs[:, -num_landmarks*2:]
    
    # Reshape landmark features to get coordinates
    predicted_landmarks = landmark_features.view(-1, num_landmarks, 2)  # Assuming each landmark has x and y coordinates
    
    return predicted_landmarks

# Function to visualize predicted landmarks on the input image
def visualize_landmarks(image_path, predicted_landmarks):
    # Read the input image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Visualize predicted landmarks on the image
    for landmark in predicted_landmarks:
        x, y = landmark  # Assuming each landmark is a tuple of (x, y) coordinates
        cv2.circle(image, (int(x), int(y)), 3, (255, 0, 0), -1)  # Draw a circle at each landmark position
    # Display the image with landmarks
    cv2.imshow('Predicted Landmarks', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Path to the trained model
best_model_path = 'C:/PROJECT_CODE/DETECTION_NET/Models/facelandmark_model.pth'

# Load the trained model
num_classes = 1
num_landmarks = 194
model = FacialLandmarkNet(num_classes, num_landmarks)
try:
    # Load the saved model state dict
    checkpoint = torch.load(best_model_path)
    # Load state dict into the model
    model.load_state_dict(checkpoint['model_state_dict'])
    # Now your model is loaded and ready for use
    model.eval()
    print("Model loaded successfully!")
except Exception as e:
    print("Error loading the model:", e)

# Preprocess input image
image_path = 'C:/PROJECT_CODE/DETECTION_NET/Helen-Images/test/305917477_2.jpg'
image_tensor = preprocess_image_inference(image_path, train_transform)

# Predict landmarks
predicted_landmarks = predict_landmarks(model, image_tensor, num_landmarks)

# Visualize predicted landmarks on the input image
visualize_landmarks(image_path, predicted_landmarks)
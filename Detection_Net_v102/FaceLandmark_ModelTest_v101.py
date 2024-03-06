import torch
import cv2
import numpy as np
from FaceLandmark_Functions_v102 import FacialLandmarkNet, val_transform
import torch
import torchvision.transforms as transforms


lr = 0.01
num_epochs = 20
best_model_path = 'C:/PROJECT_CODE/DETECTION_NET/Models/facelandmark_model.pth'

# Initialize the CNN model
num_classes = 1
num_landmarks = 194

# Load the saved model
checkpoint = torch.load(best_model_path)
model = FacialLandmarkNet(num_classes, num_landmarks)
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
    # Convert image to tensor using the validation transform
    tensor_image = val_transform(image)
    # Add batch dimension
    tensor_image = tensor_image.unsqueeze(0)
    # Forward pass through the model
    with torch.no_grad():
        outputs = model(tensor_image)
    # Extract predicted landmarks from the output tensor
    landmarks_predictions = outputs[:, :num_landmarks*2]  # Assuming landmarks are at the beginning of the output tensor
    # Reshape landmarks predictions
    landmarks_predictions = landmarks_predictions.view(-1, num_landmarks, 2)  # Reshape to match the format of the landmarks
    print(landmarks_predictions)
    return landmarks_predictions


def overlay_landmarks(image_path, landmarks):
    # Load the image using OpenCV
    image = cv2.imread(image_path)
    # Convert image to RGB (OpenCV loads images in BGR format)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Convert landmarks tensor to numpy array
    landmarks = landmarks.numpy()
    # Scale the landmarks to match the image size
    image_height, image_width, _ = image.shape
    landmarks = landmarks * np.array([image_width, image_height])
    # Draw landmarks on the image
    for landmark in landmarks:
        for point in landmark:
            x, y = int(point[0]), int(point[1])
            print(x)
            print(y)
            cv2.circle(image, (x, y), 2, (255, 0, 0), -1)  # Draw a blue circle at each landmark position
        
    # Display the image with overlaid landmarks
    cv2.imshow("Image with Landmarks", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Example usage
image_path = 'C:/PROJECT_CODE/DETECTION_NET/Helen-Images/train_1/10406776_1.jpg'
predicted_landmarks = predict_landmarks(image_path)
# overlay_landmarks(image_path, predicted_landmarks)
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau
from Tools import Image_Loader_HelenDataset_v102 as IML_Helen
from FaceLandmark_Functions_v103 import *

lr = 0.01
num_epochs = 10
best_model_path = 'C:/PROJECT_CODE/EyeNet/Models/facelandmark_model.pth'

# print(IML_Helen.train_ordered_path)
train_dataset = CustomDataset(IML_Helen.train_ordered_path, IML_Helen.train_ordered_annotation, transform=train_transform)
val_dataset = CustomDataset(IML_Helen.val_ordered_path, IML_Helen.val_ordered_annotation, transform=val_transform)

# Initialize the CNN model
num_landmarks = 194
model = FacialLandmarkNet(num_landmarks)

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
# model.train()

# Define criterion for image classification task
criterion_image = nn.CrossEntropyLoss()
# Define criterion for landmark prediction task (assuming you are using Mean Squared Error loss)
criterion_landmark = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

def run_training():
    best_val_loss = float('inf')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=120, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=120)
    for epoch in range(num_epochs):
        model.train()
        print(f'Model is in training mode: {model.training}')
        train_loss = 0.0
        # Inside your training loop function (run_training)
        for images, landmarks in train_dataloader:
            # Move data to GPU if available
            images = images.to(device)
            landmarks = landmarks.to(device)
            
            # Forward pass
            landmark_outputs = model(images)
            
            # Ensure the output shape matches the target shape
            # Example: Assuming num_landmarks is the number of landmarks
            landmark_outputs = landmark_outputs.view(-1, num_landmarks, 2)
            
            # Calculate loss
            landmark_loss = criterion_landmark(landmark_outputs, landmarks)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            landmark_loss.backward()
            optimizer.step()
            train_loss += landmark_loss.item() * images.size(0)
        
        # Validate the model
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, landmarks in val_dataloader:
                # Move data to GPU if available
                images, landmarks = images.to(device), landmarks.to(device)
                # Forward pass
                landmark_outputs = model(images)
                # Calculate landmark prediction loss (assuming using some appropriate loss function)
                landmark_loss = criterion_landmark(landmark_outputs, landmarks)  # Adjust the criterion function accordingly
                # Accumulate validation loss
                val_loss += landmark_loss.item() * images.size(0)
        
        # Update learning rate
        scheduler.step(val_loss)
        print("Learning rate:", scheduler.get_last_lr())
        
        # Print statistics
        train_loss /= len(train_dataset)
        val_loss /= len(val_dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Save the model if validation loss has improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_val_loss': best_val_loss,
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'date_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")}, 
                    best_model_path)

    print('Training complete.')

run_training()
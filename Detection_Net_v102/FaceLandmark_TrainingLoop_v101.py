
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau
from Detection_Net_Tools import Image_Loader_HelenDataset_v102 as IML_Helen
from FaceLandmark_Functions_v102 import *


lr=0.01
num_epochs = 10
best_model_path = 'C:/PROJECT_CODE/DETECTION_NET/Models/facelandmark_model.pth'

train_dataset = CustomDataset(IML_Helen.train_ordered_path, IML_Helen.train_ordered_annotation, IML_Helen.train_label, transform=train_transform)
val_dataset = CustomDataset(IML_Helen.train_ordered_path, IML_Helen.train_ordered_annotation, IML_Helen.train_label, transform=val_transform)

# Initialize the CNN model
num_classes = 1
num_landmarks = 194
model = FacialLandmarkNet(num_classes, num_landmarks)

# Define device
if torch.cuda.is_available():
    print("\nGPU ACCELERATED!\n")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

def run_training():
    best_val_loss = float('inf')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32)
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for images, landmarks, labels in train_dataloader:
            images = images.to(device)
            landmarks = landmarks.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images, landmarks)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        
        # Validate the model
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, landmarks, labels in val_dataloader:
                images, landmarks, labels = images.to(device), landmarks.to(device), labels.to(device)
                outputs = model(images, landmarks)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print statistics
        train_loss /= len(train_dataset)
        val_loss /= len(val_dataset)
        accuracy = 100 * correct / total
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {accuracy:.2f}%')
        
        # Save the model if validation loss has improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_val_loss': best_val_loss,
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'metrics': {'accuracy': accuracy},
                        'date_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")}, 
                    best_model_path)

    print('Training complete.')
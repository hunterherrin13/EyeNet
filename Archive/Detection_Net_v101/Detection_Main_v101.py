
from Detection_Net_Tools import Image_Loader_v101 as IML
from FaceModel_Functions_v103 import *


lr=0.001
num_epochs = 50
best_model_path = 'C:/PROJECT_CODE/DETECTION_NET/Models/best_model.pth'

train_image_paths,train_labels = IML.train_images,IML.encoded_names
val_image_paths,val_labels = IML.train_images,IML.encoded_names
# train_image_paths,train_labels = test_images,test_names
# val_image_paths,val_labels = test_images,test_names
train_dataset = CustomDataset(train_image_paths, train_labels, transform=train_transform)
val_dataset = CustomDataset(val_image_paths, val_labels, transform=val_transform)

# Initialize the CNN model
num_classes = len(IML.unique_names)
# model = CNNModel(num_classes)
model = DeepFaceNet(num_classes)

# Define device
if torch.cuda.is_available():
    print("\nGPU ACCELERATED!\n")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr)

def run_training():
    # Define DataLoader for training and validation sets
    best_val_loss = float('inf')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32)
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        
        # Validate the model
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # Print statistics
        train_loss /= len(train_dataset)
        val_loss /= len(val_dataset)
        accuracy = 100 * correct / total
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {accuracy:.2f}%')
        
        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': best_val_loss}, 
	    best_model_path)


    print('Training complete.')


###########################################################################
###########################################################################



# run_training()

model = DeepFaceNet(num_classes)
optimizer = optim.SGD(model.parameters(), lr, momentum=0.9)
checkpoint = torch.load(best_model_path)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
model.eval()

print(epoch)
print(loss)

for image_path in val_image_paths :
    print(image_path)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image = cv2.resize(image, (240, 240))  # Resize the image
    image = image / 255.0  # Normalize pixel values to [0, 1]
    image = np.transpose(image, (2, 0, 1))  # Transpose to (C, H, W) format
    image = torch.tensor(image).float()  # Convert numpy array to PyTorch tensor
    # Add batch dimension since the model expects a batch of images
    image = image.unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        output = model(image)
    # Get the predicted class
    _, predicted = torch.max(output, 1)
    print("Predicted class:", predicted.item())
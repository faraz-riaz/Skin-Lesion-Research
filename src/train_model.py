import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from VGG_definitions import VGG16WithAttention, ImageDataset, transform, BATCH_SIZE
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


# Set device and create results directory
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
os.makedirs('../results', exist_ok=True)

# Create dataset and dataloader
train_df = pd.read_csv(r'../images/metadata/train_metadata_updated.csv')
val_df = pd.read_csv(r'../images/metadata/val_metadata_updated.csv')

train_dataset = ImageDataset(dataframe=train_df, directory='../images/train_images', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

# Initialize model, loss function, and optimizer
model = VGG16WithAttention().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 2  # Increased number of epochs for better training
best_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    # Progress bar for training
    with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
        for images, labels in pbar:
            # Move data to device
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update running loss
            running_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Calculate average loss for the epoch
    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {epoch_loss:.4f}")
    
    # Save model if it has the best loss so far
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': best_loss,
        }, '../results/vgg16_attention_best.pth')
        print(f"Saved best model with loss: {best_loss:.4f}")

# Save final model weights
torch.save({
    'epoch': num_epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': epoch_loss,
}, '../results/vgg16_attention_final.pth')

print("Training complete")
print(f"Best loss achieved: {best_loss:.4f}")
print("Model weights saved")

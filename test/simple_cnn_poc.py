
import os
from io import BytesIO
import boto3
import math
import torch
import torch.nn as nn
import torch.optim as optim
from utils.utils.data import map_labels, collate_get_image
from utils.utils.model import save_model
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = 'simple-cnn-poc'
HF_DATASET = os.getenv('HF_DATASET')
HF_DATASET_TRAIN_SIZE = int(os.getenv('HF_DATASET_TRAIN_SIZE'))

s3 = boto3.client('s3')
 
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(16 * 112 * 112, num_classes)
        )

    def forward(self, x):
        return self.model(x)


label_converter = {'PE and Others': 1, 'PE Only': 1, 'No Finding': 0}
dataset = load_dataset(HF_DATASET, streaming=True)
# dataset = load_dataset(HF_DATASET)

# print(dataset['train'].select(range(1000)))
dataset = dataset.map(lambda x: map_labels(x, label_converter))

batch_size = 32
n_batches = math.ceil(HF_DATASET_TRAIN_SIZE/batch_size)

train_loader = DataLoader(
    # dataset['train'].select(range(1280)),
    dataset['train'],
    batch_size=batch_size,
    # shuffle = True,
    collate_fn=collate_get_image
    )


val_loader = DataLoader(
    # dataset['validation'].select(range(448)),
    dataset['validation'],
    batch_size=batch_size,
    # shuffle = False,
    collate_fn=collate_get_image
    )


# Define Model
model = SimpleCNN(num_classes=len(set(label_converter.values())))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define Loss Function
criterion = nn.CrossEntropyLoss()

# Define Optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training Loop
num_epochs = 2
for epoch in range(1, num_epochs+1):
    model.train()
    for i, batch in tqdm(enumerate(train_loader, 1)):
        images, labels = batch['images'], batch['labels']
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Save a checkpoint
        if i%50 == 0 or i == n_batches:

            print(f'Epoch [{epoch}/{num_epochs}], '
                  f'Batch [{i}/{n_batches}], '
                  f'Train Loss: {loss.item():.4f}')

            path = f"{MODEL_NAME}/checkpoints/epoch_{epoch}_batch_{i}.pth"
            data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }

            save_model(data, path)


    # Validation loop
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for i, batch in tqdm(enumerate(val_loader)):
            images, labels = batch['images'], batch['labels']
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            val_loss += criterion(outputs, labels).item()

    print(f'Epoch [{epoch}/{num_epochs}], '
          f'Train Loss: {loss.item():.4f}, '
          f'Val Loss: {val_loss/len(val_loader):.4f}, '
          f'Val Accuracy: {(100 * correct / total):.2f}%')

# Save resulting model
path = f"{MODEL_NAME}-fully-trained.pth"
data = {
    'n_epochs': num_epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}

save_model(data, path)
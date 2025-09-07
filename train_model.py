import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# 1️⃣ setting up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2️⃣ Data augmentation + preprocessing
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
])

train_dataset = datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
trainloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# 3️⃣ loading a pre-trained model
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# 4️⃣ modifying the final layer
num_classes = 10
model.fc = nn.Linear(model.fc.in_features, num_classes)

# 5️⃣ Freezing layers
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True

model.to(device)

# 6️⃣ Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

# 7️⃣ fine-tuning the model
epochs = 5
for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(trainloader)}")

# 8️⃣ saving the model
torch.save(model.state_dict(), "models/transfer_model.pth")
print("✅ Model trained and saved!")
import torch
import torchvision
from torchvision import datasets, transforms
from torch import nn, optim
from torch.utils.data import DataLoader

# Transform (resize + convert to tensor)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])

# Load dataset
dataset = datasets.ImageFolder("dataset", transform=transform)

# Create data loader
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Load pre-trained model
model = torchvision.models.resnet18(pretrained=True)

# Modify last layer (3 classes: empty, half, full)
model.fc = nn.Linear(model.fc.in_features, 3)

# Training setup
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# Training loop
for epoch in range(5):  # you can increase to 10 later
    total_loss = 0

    for images, labels in loader:
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

#  Save model
torch.save(model.state_dict(), "bin_model.pth")

print("✅ Model training complete and saved!")

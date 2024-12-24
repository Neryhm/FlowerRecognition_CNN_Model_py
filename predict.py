import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

# Define the CNN model (same architecture as the one used in training)
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))

        x = x.view(-1, 128 * 28 * 28)  # Flatten the tensor

        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)

        return x


# Load the model and its weights
model = CNNModel(num_classes=5)  # 5 classes: daisy, dandelion, rose, sunflower, tulip
model.load_state_dict(torch.load('cnn_model_weights.pth'))
model.eval()  # Set the model to evaluation mode

# Define the transformation used during training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Get the class labels
flower_classes = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# Input the image path you want to predict on
image_path = input("Enter the image path: ")

# Check if the file exists
if not os.path.exists(image_path):
    print(f"The file at {image_path} does not exist. Please check the path and try again.")
else:
    # Load and transform the image
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

    # Make a prediction
    with torch.no_grad():  # Disable gradient computation for inference
        output = model(img_tensor)  # Forward pass
        _, predicted_class = torch.max(output, 1)  # Get the predicted class index

    # Get the class label
    predicted_class_label = flower_classes[predicted_class.item()]
    print(f"Predicted class: {predicted_class_label}, Image: {image_path}")

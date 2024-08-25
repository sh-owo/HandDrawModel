import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.utils.data as data


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

        # Calculate the size of the tensor after convolutions and pooling
        self._initialize_fc_layers()

    def _initialize_fc_layers(self):
        # Create a dummy input to calculate the flattened size dynamically
        dummy_input = torch.zeros(1, 1, 28, 28)
        dummy_output = self._forward_conv(dummy_input)
        flattened_size = dummy_output.numel()

        # Define fully connected layers based on the flattened size
        self.fc1 = nn.Linear(flattened_size, 128)
        self.fc2 = nn.Linear(128, 10)

    def _forward_conv(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = torch.flatten(x, 1)  # Flatten the output for fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# Function to load the MNIST dataset
def dataset_loader(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # Normalization based on MNIST stats
    ])
    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    return trainloader


# Function to train the model
def train_model(model, trainloader, criterion, optimizer, device, epochs=10):
    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()  # Clear gradients
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Calculate loss

            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

            running_loss += loss.item()
            if i % 100 == 99:  # Print every 100 batches
                print(f'Epoch: {epoch + 1}, Batch: {i + 1}, Loss: {running_loss / 100:.4f}')
                running_loss = 0.0

    # Save the trained model
    model_path = 'model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


# Main function to execute the training
def main():
    try:
        # Initialize parameters
        batch_size = 100
        learning_rate = 0.001
        epochs = 20

        # Load dataset
        trainloader = dataset_loader(batch_size=batch_size)

        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Initialize model, criterion, and optimizer
        model = Net()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

        # Train the model
        train_model(model, trainloader, criterion, optimizer, device, epochs)

    except Exception as e:
        print(f"An error occurred: {e}")


# Run the main function
if __name__ == "__main__":
    main()

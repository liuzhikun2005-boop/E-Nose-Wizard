import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Load training and testing datasets
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# Extract features and labels from training dataset
train_features = train_data.iloc[:, :-1]  # First four columns are features
train_labels = train_data.iloc[:, -1]     # Last column is the label

# Extract features and labels from testing dataset
test_features = test_data.iloc[:, :-1]    # First four columns are features
test_labels = test_data.iloc[:, -1]       # Last column is the label

# Normalize the features
scaler = MinMaxScaler()
train_normalized_features = scaler.fit_transform(train_features)
test_normalized_features = scaler.transform(test_features)

# Convert normalized features and labels back to DataFrames
train_normalized_data = pd.DataFrame(train_normalized_features, columns=train_features.columns)
train_normalized_data['label'] = train_labels

test_normalized_data = pd.DataFrame(test_normalized_features, columns=test_features.columns)
test_normalized_data['label'] = test_labels

# Save the processed datasets to new CSV files
train_normalized_data.to_csv("normalized_train_data.csv", index=False)
test_normalized_data.to_csv("normalized_test_data.csv", index=False)

# Define the SVM model
class SVM(nn.Module):
    def __init__(self):
        super(SVM, self).__init__()
        self.linear = nn.Linear(4, 1)  # 4 input features connected to a single output neuron

    def forward(self, x):
        return self.linear(x)

# Instantiate the SVM model
model = SVM()

# Loss function: Hinge Loss
criterion = nn.HingeEmbeddingLoss()

# Optimizer: Stochastic Gradient Descent
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# Load the processed training and testing datasets
train_normalized_data = pd.read_csv("normalized_train_data.csv")
test_normalized_data = pd.read_csv("normalized_test_data.csv")

# Split features and labels for training and testing
X_train = train_normalized_data.iloc[:, :-1].values  # Features for training
y_train = train_normalized_data.iloc[:, -1].values   # Labels for training

X_test = test_normalized_data.iloc[:, :-1].values    # Features for testing
y_test = test_normalized_data.iloc[:, -1].values      # Labels for testing

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Train the model
num_epochs = 500
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs.squeeze(), y_train_tensor)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model performance on the testing dataset
with torch.no_grad():
    outputs = model(X_test_tensor)
    predicted_labels = torch.sign(outputs).squeeze().numpy()
    accuracy = (predicted_labels == y_test).mean()
    print(f'Testing Accuracy: {accuracy:.4f}')

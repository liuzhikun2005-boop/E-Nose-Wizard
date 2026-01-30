import torch
import torch.nn as nn
import pandas as pd
import tkinter as tk
from tkinter import ttk
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Load dataset
data = pd.read_csv("dataclass.csv")

# Define features and label columns
feature_cols = data.columns[:-1]
label_col = data.columns[-1]

# Define preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), feature_cols),  # Scale numeric features
        ('imputer', SimpleImputer(strategy='most_frequent'), label_col)  # Handle missing values in label column
    ])

# Split features and labels
X = data[feature_cols]
y = data[label_col]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply preprocessing pipeline to train and test data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_processed, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)

# Define SVM model
class SVM(nn.Module):
    def __init__(self):
        super(SVM, self).__init__()
        self.linear = nn.Linear(X_train_processed.shape[1], 1)

    def forward(self, x):
        return self.linear(x)

# Initialize SVM model
svm_model = SVM()

# Define loss function and optimizer
criterion = nn.HingeEmbeddingLoss()
optimizer = torch.optim.SGD(svm_model.parameters(), lr=0.01)

# Model training
num_epochs = 1000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = svm_model(X_train_tensor)
    loss = criterion(outputs.squeeze(), y_train_tensor)
    loss.backward()
    optimizer.step()

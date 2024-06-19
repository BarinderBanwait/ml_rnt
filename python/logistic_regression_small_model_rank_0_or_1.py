import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


# Load the dataframe
df = pd.read_csv("../data_files/ecq_rank_0_or_1_cond_limit_5000_one_per_isog_class_balanced.csv")
# Print the first few rows of the dataframe
print(df.head())

# Print unique values in the 'rank' column
print("Unique rank values are ", df['rank'].unique())

# Print the data types of the columns
print("Th column values are of types: ", set(df.dtypes.to_list()))

# But the columns themselves are strings!
column_names = df.columns.tolist()
column_types = set([type(name) for name in column_names])
print("The column names are of type: ", column_types)

# Divide dataframe into features and label (rank); change data types to float
X = df.drop(columns=['rank']).values.astype(np.float32)
y = df['rank'].values.astype(np.float32)

# Split the data into training and test sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)


print("The training set has ", X_train.shape[0], " examples.")
print("The test set has ", X_test.shape[0], " examples.")

# Scale the features
primes_strings = column_names[:-1]
primes = [int(primes_strings[i]) for i in range(len(primes_strings))]
def scale_by_ap(data_set):
    scaling_factor = [1/(2*np.sqrt(primes[i])) for i in range(len(primes))]
    scaling_factor = np.array(scaling_factor, dtype=np.float32)
    return data_set * scaling_factor

# scaler = StandardScaler()
X_train = scale_by_ap(X_train)
X_test = scale_by_ap(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train)
y_train_tensor = torch.tensor(y_train).view(-1, 1)  # Reshape y to be a column vector
X_test_tensor = torch.tensor(X_test)
y_test_tensor = torch.tensor(y_test).view(-1, 1)  # Reshape y to be a column vector

# Create DataLoader for training data
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# Define the logistic regression model
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

input_dim = X_train.shape[1]
model = LogisticRegressionModel(input_dim)

# Define the loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Train the model
num_epochs = 40

for epoch in range(num_epochs):
    for inputs, labels in train_dataloader:
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model on the test set
with torch.no_grad():
    predicted = model(X_test_tensor).round()
    accuracy = (predicted.eq(y_test_tensor).sum().item()) / y_test_tensor.size(0)
    print(f'Accuracy on test set: {accuracy * 100:.2f}%')

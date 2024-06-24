import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load the dataframe
df = pd.read_csv("../data_files/ecq_rank_0_or_1_cond_limit_5000_one_per_isog_class_balanced.csv")
# Print the first few rows of the dataframe
print(df.head())

# Print unique values in the 'rank' column
print("Unique rank values are ", df['rank'].unique())

# Divide dataframe into features and label (rank); change data types to float
X = df.drop(columns=['rank']).values.astype(np.float32)
y = df['rank'].values.astype(np.float32)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

# Scale the features
column_names = df.columns.tolist()
primes_strings = column_names[:-1]
primes = [int(primes_strings[i]) for i in range(len(primes_strings))]
def scale_by_ap(data_set):
    scaling_factor = [1/(2*np.sqrt(primes[i])) for i in range(len(primes))]
    scaling_factor = np.array(scaling_factor, dtype=np.float32)
    return data_set * scaling_factor

X_train = scale_by_ap(X_train)
X_test = scale_by_ap(X_test)

# Initialize and train the Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy on test set: {accuracy * 100:.2f}%')
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the training dataset
train_data = pd.read_csv('train.csv')  # Load the Titanic training dataset from a CSV file

# Explore the dataset by printing the first few rows and the dataset information
# print(train_data.head())  # Print the first 5 rows of the dataset to get an overview
print(train_data.info())  # Print a concise summary of the dataset, including column names and data types

# # Preprocess the data by dropping irrelevant columns and removing rows with missing values
# train_data = train_data.drop(['Name', 'Ticket', 'Cabin'], axis=1)  # Drop columns that are not useful for prediction
# train_data = train_data.dropna()  # Remove rows with missing values to ensure the dataset is complete

# # Convert categorical variables ('Sex' and 'Embarked') to numeric values for model compatibility
# train_data['Sex'] = train_data['Sex'].map({'male': 0, 'female': 1})  # Convert 'Sex' to numeric values: male=0, female=1
# train_data['Embarked'] = train_data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})  # Convert 'Embarked' to numeric values

# # Define the features (X) and the target variable (y) for the model
# X = train_data.drop('Survived', axis=1)  # Features: all columns except 'Survived'
# y = train_data['Survived']  # Target variable: 'Survived'

# # Split the data into training and testing sets to evaluate the model's performance
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 80% training, 20% testing

# # Build a logistic regression model using the training data
# model = LogisticRegression()  # Initialize the logistic regression model
# model.fit(X_train, y_train)  # Train the model using the training data

# # Make predictions on the testing data
# y_pred = model.predict(X_test)  # Predict the target variable for the test set

# # Evaluate the model's accuracy by comparing the predicted values with the actual values
# accuracy = accuracy_score(y_test, y_pred)  # Calculate the accuracy of the model
# print(f'Accuracy: {accuracy}')  # Print the accuracy of the mo
#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# In[3]:


# Load the datasets
train_df = pd.read_csv("E:/Projects/Encryptix/Credit Card/fraudTrain.csv")
test_df = pd.read_csv("E:/Projects/Encryptix/Credit Card/fraudTest.csv")

# Display the first few rows and data types of the training dataset
print("Training Data:")
print(train_df.head())
print("Data Types:")
print(train_df.dtypes)


# In[4]:


# Identify and convert date-time columns in both train and test datasets
date_time_columns = [col for col in train_df.columns if train_df[col].dtype == 'object' and 'date' in col.lower()]

for col in date_time_columns:
    train_df[col] = pd.to_datetime(train_df[col], errors='coerce')
    test_df[col] = pd.to_datetime(test_df[col], errors='coerce')

    # Extract useful features from the datetime column
    train_df[col + '_year'] = train_df[col].dt.year
    train_df[col + '_month'] = train_df[col].dt.month
    train_df[col + '_day'] = train_df[col].dt.day
    train_df[col + '_hour'] = train_df[col].dt.hour
    
    test_df[col + '_year'] = test_df[col].dt.year
    test_df[col + '_month'] = test_df[col].dt.month
    test_df[col + '_day'] = test_df[col].dt.day
    test_df[col + '_hour'] = test_df[col].dt.hour
    
    # Drop the original date-time column
    train_df = train_df.drop(col, axis=1)
    test_df = test_df.drop(col, axis=1)

# Convert categorical columns to numeric using pd.Categorical
categorical_columns = train_df.select_dtypes(include=['object']).columns

for col in categorical_columns:
    # Combine train and test data to ensure consistent categories
    combined = pd.concat([train_df[col], test_df[col]], axis=0)
    categories = pd.Categorical(combined).categories
    
    # Apply the same categories to train and test data
    train_df[col] = pd.Categorical(train_df[col], categories=categories).codes
    test_df[col] = pd.Categorical(test_df[col], categories=categories).codes

# Define features and target variable for training
target_column = 'is_fraud'  # Ensure this is the correct column name

X_train = train_df.drop(target_column, axis=1)
y_train = train_df[target_column]

# Handle missing values if needed
X_train.fillna(method='ffill', inplace=True)
test_df.fillna(method='ffill', inplace=True)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(test_df.drop(target_column, axis=1))


# In[5]:


# Split into training and validation sets
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=42)

# Train models
log_reg = LogisticRegression()
log_reg.fit(X_train_split, y_train_split)

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train_split, y_train_split)


# In[8]:


# Evaluate models on validation set
print("Logistic Regression Classification Report:")
y_pred_log_reg = log_reg.predict(X_val_split)
print(classification_report(y_val_split, y_pred_log_reg))

# Make predictions on the test dataset
test_predictions_log_reg = log_reg.predict(X_test_scaled)

# Add predictions to test dataframe
test_df['Logistic_Regression_Pred'] = test_predictions_log_reg

# Display fraud detections
print("Fraudulent Transactions Detected by Logistic Regression:")
print(test_df[test_df['Logistic_Regression_Pred'] == 1].head())


# In[9]:


# Evaluate models on validation set
print("Decision Tree Classification Report:")
y_pred_decision_tree = decision_tree.predict(X_val_split)
print(classification_report(y_val_split, y_pred_decision_tree))
# Make predictions on the test dataset
test_predictions_decision_tree = decision_tree.predict(X_test_scaled)
# Add predictions to test dataframe
test_df['Decision_Tree_Pred'] = test_predictions_decision_tree
# Display fraud detections
print("Fraudulent Transactions Detected by Decision Tree:")
print(test_df[test_df['Decision_Tree_Pred'] == 1].head())


# In[ ]:





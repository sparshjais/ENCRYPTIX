#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[11]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


# # Load and Clean the Dataset

# In[12]:


data = pd.read_csv("spam.csv", encoding='latin-1')
data = data[['v1', 'v2']]
data.columns = ['label', 'message']


# # Train the Model

# In[13]:


X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer(min_df=1, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train the Regression Model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Evaluate the Model
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n", report)


# # Test the Model

# In[4]:


def classify_message(message):
    message_tfidf = vectorizer.transform([message])
    prediction = model.predict(message_tfidf)
    return prediction[0]

print("Interactive SMS Spam Detection Model")
print("Type 'exit' to stop.")

while True:
    user_input = input("\nEnter a message to classify: ")
    if user_input.lower() == 'exit':
        break
    prediction = classify_message(user_input)
    print(f"The message is classified as: {prediction}")

print("Thank you for using the model!")


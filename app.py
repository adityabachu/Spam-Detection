import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Function to preprocess text data
def clean_text(text):
    # Implement your preprocessing steps here
    return text

# Load data
@st.cache
def load_data():
    df = pd.read_csv("SMSSpamCollection.tsv", sep="\t", names=["label", "text"])
    return df

# Preprocess data
def preprocess_data(df):
    df['cleaned_text'] = df['text'].apply(clean_text)
    return df

# Train model
def train_model(X_train, y_train):
    classifier = GaussianNB()
    classifier.fit(X_train.toarray(), y_train)
    return classifier

# Evaluate model
def evaluate_model(classifier, X_test, y_test):
    y_pred = classifier.predict(X_test.toarray())
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return accuracy, cm

# Define Streamlit app
def main():
    st.title("Spam Detection App")

    # Load data
    df = load_data()

    # Preprocess data
    df = preprocess_data(df)

    # Split data into features and target
    X = df['cleaned_text']
    y = df['label']

    # Apply CountVectorizer
    count_vect = CountVectorizer()
    X_counts = count_vect.fit_transform(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_counts, y, test_size=0.2, random_state=42)

    # Train model
    classifier = train_model(X_train, y_train)

    # Evaluate model
    accuracy, cm = evaluate_model(classifier, X_test, y_test)

    # Display model performance
    st.subheader("Model Performance")
    st.write("Accuracy:", accuracy)
    st.write("Confusion Matrix:")
    st.write(cm)

    # Display prediction form
    st.subheader("Predictions")
    message = st.text_input("Enter a message:")
    if st.button("Predict"):
        # Preprocess input message
        cleaned_message = clean_text(message)
        # Transform message using CountVectorizer
        message_vectorized = count_vect.transform([cleaned_message])
        # Make prediction
        prediction = classifier.predict(message_vectorized.toarray())[0]
        st.write("Predicted:", prediction)

if __name__ == "__main__":
    main()

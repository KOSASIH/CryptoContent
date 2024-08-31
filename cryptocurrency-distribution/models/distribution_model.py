"""
Distribution Model

This module provides a machine learning model for predicting cryptocurrency distribution based on various features.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

class DistributionModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()

    def train(self, data):
        """
        Train the distribution model on the given data.

        :param data: Pandas DataFrame containing features and target variable
        :return: Trained model
        """
        X = data.drop("distribution", axis=1)
        y = data["distribution"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Handle class imbalance using SMOTE
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        # Scale features using StandardScaler
        X_train_scaled = self.scaler.fit_transform(X_train_res)
        X_test_scaled = self.scaler.transform(X_test)

        self.model.fit(X_train_scaled, y_train_res)
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.2f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        return self.model

    def predict(self, data):
        """
        Predict cryptocurrency distribution based on the given features.

        :param data: Pandas DataFrame containing features
        :return: Predicted distribution
        """
        # Scale features using StandardScaler
        data_scaled = self.scaler.transform(data)
        return self.model.predict(data_scaled)

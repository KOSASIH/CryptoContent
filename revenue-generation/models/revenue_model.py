"""
Revenue Model

This module provides a machine learning model for predicting revenue based on various features.
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

class RevenueModel:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()

    def train(self, data):
        """
        Train the revenue model on the given data.

        :param data: Pandas DataFrame containing features and target variable
        :return: Trained model
        """
        X = data.drop("revenue", axis=1)
        y = data["revenue"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale features using StandardScaler
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.model.fit(X_train_scaled, y_train)
        y_pred = self.model.predict(X_test_scaled)
        score = mean_squared_error(y_test, y_pred, squared=False)
        print(f"RMSE: {score:.2f}")
        return self.model

    def predict(self, data):
        """
        Predict revenue based on the given features.

        :param data: Pandas DataFrame containing features
        :return: Predicted revenue
        """
        # Scale features using StandardScaler
        data_scaled = self.scaler.transform(data)
        return self.model.predict(data_scaled)

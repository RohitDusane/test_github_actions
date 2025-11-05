import pandas as pd
import numpy as np
import pytest
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score

# data
from sklearn.datasets import load_iris
import joblib

def load_data():
    try:
        df = load_iris()
        X = df.data
        y = df.target
        return X, y
    except Exception as e:
        print('Failed to load data', e)

def preprocess_data(X, y):
    return train_test_split(X,y, test_size= 0.15, random_state=24)

def train_model(X_train,y_train):
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    return dt

def eval_model(dt, X_test, y_test):
    y_pred = dt.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')  # Weighted to handle multi-class
    return accuracy, f1

def main():
    X,y = load_data()
    X_train,X_test,y_train,y_test = preprocess_data(X,y)
    model = train_model(X_train,y_train)
    acc, f1s = eval_model(model, X_test,y_test)
    print(f"Model Accuracy: {acc * 100:.2f}%")
    print(f"Model F1-Score: {f1s * 100:.2f}%")
    return model

if __name__=='__main__':
    main()

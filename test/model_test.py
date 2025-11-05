import pytest
from model.model import load_data, preprocess_data, train_model, evaluate_model

def test_data_loading():
    X, y = load_data()
    assert X.shape[0] > 0  # Ensure there are data points
    assert y.shape[0] == X.shape[0]  # Ensure features and labels match

def test_model_accuracy():
    X, y = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    model = train_model(X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test)
    assert accuracy > 0.9  # Ensure the model achieves above 90% accuracy (reasonable threshold for Iris dataset)

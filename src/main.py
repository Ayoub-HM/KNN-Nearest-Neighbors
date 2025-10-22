from sklearn.datasets import load_iris
from knn import KNN
from metrics import accuracy_score

def main():
    # Load the Iris dataset
    X, y = load_iris(return_X_y=True)
    print(f"Shape X: {X.shape}, Shape y: {y.shape}")

    # Create and train the model
    knn = KNN(k=5)
    knn.fit(X, y)

    # Predict
    y_pred = knn.predict(X)

    # Evaluate accuracy
    acc = accuracy_score(y, y_pred)
    print(f"Accuracy: {acc:.2f}")

if __name__ == "__main__":
    main()

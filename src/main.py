from sklearn.datasets import load_iris
from knn import KNN
from metrics import accuracy_score


def search_best_k(X, y, k_min=1, k_max=20):
    """Try several k values and return the best one with all scores."""
    scores = {}

    for k in range(k_min, k_max + 1):
        knn = KNN(k=k)
        knn.fit(X, y)
        y_pred = knn.predict(X)
        scores[k] = accuracy_score(y, y_pred)

    best_k = max(scores, key=scores.get)
    return best_k, scores


def main():
    # Load the Iris dataset
    X, y = load_iris(return_X_y=True)
    print(f"Shape X: {X.shape}, Shape y: {y.shape}")

    # Find the best k between 1 and 20
    best_k, scores = search_best_k(X, y, k_min=1, k_max=20)

    print("\nAccuracy by k:")
    for k, score in scores.items():
        print(f"k={k:2d} -> accuracy={score:.2f}")

    print(f"\nBest k: {best_k} (accuracy={scores[best_k]:.2f})")

if __name__ == "__main__":
    main()

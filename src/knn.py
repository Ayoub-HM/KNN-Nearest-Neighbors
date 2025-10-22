from distances import get_distances, get_k_closest
from statistics import mode

class KNN:
    def __init__(self, k):
        self.k = k
        self.X = None
        self.y = None

    def fit(self, X_fit, y_fit):
        """Store the training data and labels."""
        self.X = X_fit
        self.y = y_fit

    def _predict_single_x(self, x):
        """Predict the label for a single sample."""
        distances = get_distances(x, self.X)
        k_indices = get_k_closest(distances, self.k)
        k_labels = [self.y[i] for i in k_indices]
        return mode(k_labels)

    def predict(self, X):
        """Predict the labels for a list of samples."""
        return [self._predict_single_x(x) for x in X]

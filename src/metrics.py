"""metrics.py

Module containing metrics utility functions"""

# src/metrics.py
def accuracy_score(y_true, y_pred):
    """Calcule le pourcentage de prédictions correctes."""
    correct = sum(yt == yp for yt, yp in zip(y_true, y_pred))
    return correct / len(y_true)

import math as mt

def get_distance(p, q):
    """Return the Euclidean distance between two points p and q."""
    return mt.sqrt(sum((pi - qi) ** 2 for pi, qi in zip(p, q)))

def get_distances(p, X):
    """Return the list of distances between point p and each point in X."""
    distances = [get_distance(p, xi) for xi in X]
    return distances

def get_k_closest(D, K):
    """Return the indices of the K smallest values in the list D."""
    return sorted(range(len(D)), key=lambda i: D[i])[:K]

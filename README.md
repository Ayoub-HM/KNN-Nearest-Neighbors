# KNN-Nearest-Neighbors
KNN-Nearest-Neighbors
# Exercice 1 - Nearest Neighbors

---

![](https://moblobi.com/wp-content/uploads/2018/07/ev-3d-yazici-moblobi.jpg)

## Objective

Code a first Machine Learning algorithm from scratch.

## Instructions

In order to introduce Machine Learning and how it works, today we will implement from scratch the **K-Nearest Neighbors algorithm**.

The coding will be done in `./src/`.

![](http://cs231n.github.io/assets/knn.jpeg)

### 1. Compute distances

Let's start by creating **distances** functions that will be useful for implementing the kNN algorithm.

Open the `distances.py` file and implement the 3 following functions according to the instructions below:

* `get_distance(p, q)`
* `get_distances(x, X)`
* `get_k_closest(D, K)`

Those functions will be useful for building our kNN model.

#### `get_distance(p, q)`

First, let's create the function `get_distance` that corresponds to the [euclidean distance](https://en.wikipedia.org/wiki/Euclidean_distance) between 2 points p and q

This function takes 2 points (tuples) `p` and `q` as arguments, and return the euclidean distance between them.

`p` and `q` are tuples and the function should return a float.

#### `get_distances(p, X)`

`X` is a list of points and this function `get_distances` should return the list of distances between `p` and each point of `X`.

It takes a point (tuple) `p` and a list of points `X` (containing points with same dimensions as `p`) as arguments, and it returns a list of float.

### 2. Get closest points

#### `get_k_closest(l, K)`

Now create a new function that takes a list of values `l` and an integer `K`. It will return the **indices** of the K smallest values within the list.

We call it `get_k_closest` as we will apply this function to a list of distances, and by return the smallest values it will return the index of the closest points.

### 3. Implement kNN class

Now that we have built some utility functions, let's implement our **model as a class**.

Open the file `knn.py` and implement the class `KNN`.

1. Build the class structure with the following empty methods `__init__`, `fit`, `_predict_single_x` and `predict`.
2. Implement the method `fit` for the `KNN` class, this method should be
   responsible of training the model. It takes 2 arguments:
   `X` and `y` where `X` is a list of data points, and `y` a list of targets.

> 🔦 **Hint**: Unlike most algorithms, k-NN does nothing at training (when you fit). It just memorizes the data points and the class associated to those points. All the work will be done during the prediction.

3. Implement the `__init__` method.  Again, not much is happening in `__init__`, it just initializes the kNN object and saves the K passed as parameter.

### 4. Retrieve predictions

Now that we can train the model using the `fit` method, we are now going to use
that model to make predictions for a single data point.

Open `knn.py` and implement the method `_predict_single_point`.

This method should take one argument, `x`, a tuple and return the majority vote for the k-closest data points.

> 🔦 *Hint*: Have a look at `mode` function from `statistics` library in order to implement the majority vote.

Let's now use our trained model and the new method `_predict_single_point` in order to predict on a set of multiple data points.

Open `knn.py` file and implement the method `predict`. This method should take one argument, `X`, a list/array of tuples and return the list of majority votes for each data point.

### 5. Accuracy score

Congrats! You built the kNN class. Now we can create a k-NN algorithm by instantiating a KNN object.

But what if we want to evaluate the performance of our model? Let's implement a function returning the accuracy of our model.

Open `metrics.py` file and implement the `accuracy_score` function. This function should:

* take 2 arguments: a list of targets and a list of predictions
* return the accuracy score, e.g. the percentage of targets that were correctly predicted

Once that you have implemented this function, it will be time to test your model.

### 6. `main.py` script

Open `main.py` and implement a script that will :

* load the iris dataset
* initiate a kNN model,
* fit it to your data
* retrieve predictions of your model on your data
* and evaluate the accuracy.

> 🔦 **Hint**: **Load Iris dataset**
>
>In order to access the iris dataset, you can use scikit-learn library that comes with pre-built functions:
>
> ``` python
> from sklearn.datasets import load_iris
> X, y = load_iris(return_X_y=True)
> ```
>
> where `X` is a vector containing your flowers features (charateristics) and `y` is a vector containing the targets (the class/type of the flowers).

**Remarks**:
- In order to execute your script written in `main.py` you can simply run (if you are in `01-Nearest-Neighbors` directory):
``` python
python src/main.py
```
- Inside your `main` script, you can print out the shape of `X` and `y`. This should print out a shape of (150, 4) for `X` and (150, ) for `y`.

### 8. Conclusion

Congratulations! 🎉

You are now ready to play! Import your `kNN` model,  make predictions and evaluate your accuracy with `accuracy_score` function.

If you have some time left, address the following bonus questions.

### 9. Bonus

- Find the optimal value of k that maximizes your model's accuracy. Do you see anything wrong with this approach? Visualize it on a chart.

- Add the `metric` optional parameter, giving the possibility to use [Manhattan distance](https://en.wikipedia.org/wiki/Taxicab_geometry) instead of the Euclidian distance when computing distances. When do you think it could be more adequate to use this type of distance?

- Plot the results by drawing the data points color-coded by class and by drawing the decision boundaries of your model. Display side by side, decision boundaries for different values of k.

- Using your newly created kNN class, fit a kNN algorithm on Titanic data we previously explored in order to predict if a passenger survived or not.

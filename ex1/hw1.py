import numpy as np
import itertools

np.random.seed(42)

def preprocess(X, y):
    """
    Perform mean normalization on the features and divide the true labels by
    the range of the column.

    Input:
    - X: Inputs  (n features over m instances).
    - y: True labels.

    Returns  two values:
    - X: The mean normalized inputs.
    - y: The scaled labels.
    """
    X = np.subtract(X,np.average(X, axis=0)) / np.subtract(np.max(X, axis=0), np.min(X, axis=0))
    y = (y - np.average(y)) / (np.max(y) - np.min(y))
    return X, y

def compute_cost(X, y, theta):
    """
    Computes the average squared difference between an observation’s actual and
    predicted values for linear regression.

    Input:
    - X: Inputs  (n features over m instances).
    - y: True labels (1 value over m instances).
    - theta: The parameters (weights) of the model being learned.

    Returns a single value:
    - J: the cost associated with the current set of parameters (single number).
    """
    m = X.shape[0]
    errorMat = np.subtract(np.dot(X, theta), y)
    meanSquares = np.square(errorMat)
    J = meanSquares.sum(axis=0)
    J = J / (2*m)
    return J


def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of the model using gradient descent. Gradient descent
    is an optimization algorithm used to minimize some (loss) function by
    iteratively moving in the direction of steepest descent as defined by the
    negative of the gradient. We use gradient descent to update the parameters
    (weights) of our model.

    Input:
    - X: Inputs  (n features over m instances).
    - y: True labels (1 value over m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns two values:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    J_history = []  # Use a python list to save cost in every iteration
    m = X.shape[0]
    for i in range(num_iters):
        sum = np.dot((np.dot(X, theta) - y), X)/m
        theta = theta - np.multiply(sum,alpha)
        J_history.append(compute_cost(X, y, theta))

    return theta, J_history

def pinv(X, y):
    """
    Calculate the optimal values of the parameters using the pseudoinverse
    approach as you saw in class.

    Input:
    - X: Inputs  (n features over m instances).
    - y: True labels (1 value over m instances).

    Returns two values:
    - theta: The optimal parameters of your model.

    ########## DO NOT USE numpy.pinv ##############
    """

    #xTranspose = X.T
    pinvX1 = np.linalg.inv(np.matmul(X.T, X))
    pinvX2 = np.matmul(pinvX1,X.T)
    pinv_theta = np.matmul(pinvX2, y)
    return pinv_theta


def efficient_gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of your model, but stop the learning process once
    the improvement of the loss value is smaller than 1e-8. This function is
    very similar to the gradient descent function you already implemented.

    Input:
    - X: Inputs  (n features over m instances).
    - y: True labels (1 value over m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns two values:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """

    J_history = []  # Use a python list to save cost in every iteration
    J_iterationsHistory = []
    m = X.shape[0]
    i = 0
    while i < num_iters:
        sum1 = np.subtract(np.dot(X, theta), y)
        sum2 = np.dot(sum1, X)/m
        theta = theta - np.multiply(sum2, alpha)
        newCost = compute_cost(X, y, theta)
        if i >= 1 and J_history[i - 1] - newCost < 1e-8:
            J_history.append(newCost)
            break
        elif i % 100 == 0:
            J_history.append(newCost)
            if len(J_iterationsHistory) >= 2 and newCost > J_iterationsHistory[-1]:
                J_iterationsHistory.append(newCost)
                break
        else:
            J_history.append(newCost)
        i += 1
    return theta, J_history
# 

def find_best_alpha(X, y, iterations):
    """
    Iterate over provided values of alpha and maintain a python dictionary
    with alpha as the key and the final loss as the value.

    Input:
    - X: a dataframe that contains all relevant features.

    Returns:
    - alpha_dict: A python dictionary that hold the loss value after training
    for every value of alpha.
    """

    alphas = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 3]
    alpha_dict = {}
    for alpha in alphas:
        np.random.seed(42)
        theta = np.random.random(size=2)
        J_history = efficient_gradient_descent(X, y, theta, alpha, iterations)[1]
        lastCost = J_history[-1]
        alpha_dict[alpha] = lastCost

    return alpha_dict


def generate_triplets(X):
    """
    generate all possible sets of three features out of all relevant features
    available from the given dataset X. You might want to use the itertools
    python library.

    Input:
    - X: a dataframe that contains all relevant features.

    Returns:
    - A python list containing all feature triplets.
    """

    triplets = list(itertools.combinations(X, 3))
    return triplets


def find_best_triplet(df, triplets, alpha, num_iter):
    """
    Iterate over all possible triplets and find the triplet that best
    minimizes the cost function. For better performance, you should use the
    efficient implementation of gradient descent. You should first preprocess
    the data and obtain a array containing the columns corresponding to the
    triplet. Don't forget the bias trick.

    Input:
    - df: A dataframe that contains the data
    - triplets: a list of three strings representing three features in X.
    - alpha: The value of the best alpha previously found.
    - num_iters: The number of updates performed.

    Returns:
    - The best triplet as a python list holding the best triplet as strings.
    """
    features = df.drop(columns=['id', 'date', 'price'])
    y = np.array(df['price'])
    features, y = preprocess(features, y)
    tripletCostsDict = {}
    for triplet in triplets:
        tripletData = features[list(triplet)].values
        # bias trick
        tripletData = tripletData.T
        biasCol = np.ones_like(tripletData[0])
        tripletData = np.vstack((biasCol, tripletData)).T
        startingTheta = np.ones(shape=len(tripletData[0]))
        tripletCost = efficient_gradient_descent(tripletData, y, startingTheta, alpha, num_iter)[1][-1]
        tripletCostsDict[triplet] = tripletCost
    bestTriplet = min(tripletCostsDict, key=lambda x: tripletCostsDict.get(x))
    return bestTriplet
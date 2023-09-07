import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self, lr=0.01, n_iter=1000):
        self.lr = lr
        self.n_iter = n_iter
        self.w = 1
        self.b = 0
        self.L_prevs = np.zeros((n_iter, 1))

    def train_test_split(self, X, y, test_size=0.2, random_seed=69):
        rng = np.random.RandomState(int(random_seed))
        rng.shuffle(X)
        rng.shuffle(y)

        # Split the data.
        n_test = int(test_size * len(X)) * (-1)
        X_train, X_test = X[:n_test], X[n_test:]
        y_train, y_test = y[:n_test], y[n_test:]

        return X_train, X_test, y_train, y_test

    def fit(self, X, y):
        for i in range(self.n_iter):
            resid = np.dot(X, self.w) - y
            self.L_prevs[i] = 0.5 * np.sum(np.square(resid))
            self.b -= self.lr*np.sum(resid)
            self.w -= self.lr*np.sum(np.multiply(resid, X)) 

    def predict(self, X):
        return np.dot(X, self.w)

    def visualize(self, X, y):
        # Plot the data points and regression line
        plt.scatter(X, y)
        y_pred = self.predict(X)
        plt.plot(X, y_pred, color='red')
        plt.show()

    def MSE(self, X, y):
        y_pred = self.predict(X)
        mse = np.mean((y_pred - y) ** 2)
        return mse

    def RMSE(self, X, y):
        y_pred = self.predict(X)
        rmse = np.sqrt(np.mean((y_pred - y) ** 2))
        return rmse

    def MAE(self, X, y):
        y_pred = self.predict(X)
        mae = np.mean(np.abs(y_pred - y))
        return mae

    def MAPE(self, X, y):
        y_pred = self.predict(X)
        mape = np.mean(np.abs((y_pred - y) / y))
        return mape

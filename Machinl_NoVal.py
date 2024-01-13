import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt


class LRTemplate(object):

    def __init__(self, in_shape, w=None, lr=0.01):
        self.in_shape = in_shape
        self.w = w #if w is not None else np.zeros((in_shape, 1))
        self.lr = lr

    def cost(self, x, y):
        
        y_hat = np.dot(x, self.w)
        error = y - y_hat
        cost = np.mean(error ** 2)
        return cost

    def train(self, x, y):
        
        y_hat = np.dot(x, self.w)
        error = y - y_hat
        gradient = np.dot(x.T, error) / len(y)
        self.w += self.lr * gradient
        cost = self.cost(x, y)
        return cost

    def predict(self, x):
        
        return np.dot(x, self.w)

    def params(self):
        
        return self.w


def validate(model, X_val, y_val):
    
    y_pred = model.predict(X_val)

    
    val_error = np.mean((y_pred - y_val) ** 2)
    return val_error



df = pd.read_csv('HW_data.csv')


X = df['x'].values
y = df['y'].values


n = len(df)


train_size = int(n * 0.7)
# val_size = int(n * 0.15)


X_train = X[:train_size].reshape(-1, 1)
y_train = y[:train_size].reshape(-1, 1)
X_train = np.c_[np.ones_like(X_train), X_train]
y_train = np.c_[y_train]

# X_val = X[train_size:train_size+val_size].reshape(-1, 1)
# y_val = y[train_size:train_size+val_size].reshape(-1, 1)
# X_val = np.c_[np.ones_like(X_val), X_val]
# y_val = np.c_[y_val]

X_test = X[train_size:].reshape(-1, 1)
y_test = y[train_size:].reshape(-1, 1)
X_test = np.c_[np.ones_like(X_test), X_test]
y_test = np.c_[y_test]


# print('Shape of X_train:', X_train.shape)
# print('Shape of y_train:', y_train.shape)
# print('Shape of X_val:', X_val.shape)
# print('Shape of y_val:', y_val.shape)
# print('Shape of X_test:', X_test.shape)
# print('Shape of y_test:', y_test.shape)

w = np.array([0, -2], dtype=np.float64).reshape(-1, 1)

model = LRTemplate(in_shape=2, w=w, lr=0.01)


for i in range(1000):
    train_cost = model.train(X_train, y_train)
    if i % 100 == 0:
        print('Epoch:', i, 'Training cost:', train_cost)


y_pred = model.predict(X_train)
# plt.scatter(X_train[:, 1], y_train)
# plt.plot(X_train[:, 1], y_pred, color='red')
# plt.figure()




y_pred_test = model.predict(X_test)
# plt.scatter(X_test[:, 1], y_test)
# plt.plot(X_test[:, 1], y_pred_test, color='red')
# plt.figure()


# y_pred_valid = model.predict(X_val)
# plt.scatter(X_val[:, 1], y_val)
# plt.plot(X_val[:, 1], y_pred_valid, color='red')
# plt.figure()

test_error = np.mean((y_pred - y_train) ** 2)
print('Testing error:', test_error)

# fig, axs =plt.subplots(1,3,figsize=(15,5))
# plt.axis('equal')

plt.scatter(X_train[:, 1], y_train)
# axs[0].plot(X_train[:, 1], y_train)
# axs[0].plot(X_train[:, 1], y_pred, color='red')
plt.plot(X_train[:, 1], y_pred, color='red')

plt.scatter(X_test[:, 1], y_test)
# axs[1].plot(X_test[:, 1], y_test)
# axs[1].plot(X_test[:, 1], y_pred_test, color='red')
plt.plot(X_test[:, 1], y_pred_test, color='red')

# plt.scatter(X_val[:, 1], y_val)
# axs[1].plot(X_val[:, 1], y_val)
# axs[1].plot(X_val[:, 1], y_pred_valid, color='red')
# plt.plot(X_val[:, 1], y_pred_valid, color='red')

plt.show()
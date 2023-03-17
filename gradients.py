import numpy as np

# linear regression function = weights * input
# f = w * x

# e.g. 
# f = 2 * x
X = np.array([1, 2, 3, 4], dtype=np.float32) # inputs
Y = np.array([2, 4, 6, 8], dtype=np.float32) # correct outputs

# initialise weights with 0
w = 0.0

# model prediction
def forward(x):
    return w * x

# loss = Mean Squared Error 
def loss(y, y_predicted):
    return ((y_predicted - y)**2).mean()

# gradient
# MSE = 1/N * (w*x - y)**2
# calc derivative 
# dJ/dw = 1/N 2x (w*x - y) 
def gradient(x, y, y_predicted):
    return np.dot(2*x, y_predicted - y).mean()

print(f'Prediction before train f(5) = {forward(5):.3f}')

lr = 0.01
n_iters = 10

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = forward(X)

    # loss
    l = loss(Y, y_pred)

    # gradients
    dw = gradient(X, Y, y_pred)

    # update weights
    w -= lr * dw

    if epoch % 1 == 0:
        print(f'Epoch: {epoch + 1} Weight: {w:.3f} Loss: {l:.8f}')

print(f'Prediction after training: f(5) = {forward(5):.3f}')

'''
Output: 

Prediction before train f(5) = 0.000
Epoch: 1 Weight: 1.200 Loss: 30.00000000
Epoch: 2 Weight: 1.680 Loss: 4.79999924
Epoch: 3 Weight: 1.872 Loss: 0.76800019
Epoch: 4 Weight: 1.949 Loss: 0.12288000
Epoch: 5 Weight: 1.980 Loss: 0.01966083
Epoch: 6 Weight: 1.992 Loss: 0.00314570
Epoch: 7 Weight: 1.997 Loss: 0.00050332
Epoch: 8 Weight: 1.999 Loss: 0.00008053
Epoch: 9 Weight: 1.999 Loss: 0.00001288
Epoch: 10 Weight: 2.000 Loss: 0.00000206
Prediction after training: f(5) = 9.999
'''
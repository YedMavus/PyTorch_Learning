

import numpy as np
# f = w * x
# f = 2*x
X = np.array([1,2,3,4], dtype=np.float32)
y = np.array([2,4,6,8],dtype=np.float32)

w = 0.0

#model pred
def forward(X):
    return w*X
# loss = MSE
def loss (y,y_predicted):
    return ((y_predicted-y)**2).mean()

#gradient
#MSE = 1/N * (x*w -y)**2

def gradient(X,y,y_predicted):
    return np.dot(2*X,y_predicted-y).mean()
print(f'Prediction before training: f(5) = {forward(5):.3f}')
#training
learning_rate = 0.01
n_iters = 20
for epoch in range(n_iters):
    #preds = forqaed pass
    y_pred = forward(X)

    #loss 
    l = loss(y,y_pred)
    #gradients

    dw = gradient(X, y, y_pred)
    
    #update weights
    w -= learning_rate * dw
    if epoch %1 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:0.8f}')
print(f'Prediction after training: f(5) = {forward(5):.3f}')
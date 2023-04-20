# 1. Design model (input, output,, forward pass with all different op and layers)
# 2. Construct loss and optimiser
# training loop
#   - forward pass - computer predicton
#   - backward pass - find error in prediction and find gradients for weights
#   - update weights
import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 0. prepare data
bc = datasets.load_breast_cancer()
X,y = bc.data, bc.target
n_samples, n_features = X.shape

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=123)
sc = StandardScaler() #make data zeromean and 1 stddev
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# x_test only needs to be transformed, the scaling model doesnt need to learn from the test set

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train=y_train.view(y_train.shape[0],1) #converting row vector to column vector
y_test = y_test.view(y_test.shape[0],1)

# model
# f = wx + b, sigmoid at the end
class LogisticRegression(nn.Module):
    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)
    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted
model = LogisticRegression(n_features)

#loss and optimizer
criterion = nn.BCELoss()
learing_rate=0.1
optimizer = torch.optim.SGD(model.parameters(), lr = learing_rate)

# training loop
n_epochs = 200
for epoch in range(n_epochs):
    #forward pass and loss
    y_predicted = model(X_train)
    loss = criterion(y_predicted, y_train)
    
    #backward pass
    loss.backward()
    
    #updates
    optimizer.step()
    
    # empty grad
    optimizer.zero_grad()
    
    if(epoch+1)%10 == 0:
        print(f'epochs:{epoch+1}, loss = {loss.item():.4f}')



#evaluation of model
with torch.no_grad():
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f'accuracy = {acc:.4f}')




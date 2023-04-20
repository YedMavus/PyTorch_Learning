
import torch
import torch.nn as nn

# Here we notice that the pytorch backpropagation algo performs worse than the manually defined model - more number of epochs required for training..

# 1. Design model (input, output,, forward pass with all different op and layers)
# 2. Construct loss and optimiser
# training loop
#   - forward pass - computer predicton
#   - backward pass - find error in prediction and find gradients for weights
#   - update weights

# f = w * x
# f = 2*x

# The shaped is changed since nn.Linear is used
X = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)
y = torch.tensor([[2],[4],[6],[8]],dtype=torch.float32)

X_test = torch.tensor([5],dtype=torch.float32)
n_samples, n_features = X.shape
input_size = n_features
output_size = n_features


print(n_samples,n_features)
model = nn.Linear(input_size,output_size)
# For custom linear reg model or any other model

class LinearRegression(nn.Module):
    def __init__(self,input_dim, output_dim):
        super(LinearRegression, self).__init__()
        #define layers
        self.lin = nn.Linear(input_dim, output_dim)
    def forward(self,x):
        return self.lin(x)
model = LinearRegression(input_size,output_size)


# w = torch.tensor(0.0, dtype=torch.float32, requires_grad= True)


# #model pred
# def forward(X):
#     return w*X




## loss = MSE
# # Manually defined loss
# def loss (y,y_predicted):
#     return ((y_predicted-y)**2).mean()



#gradient
#MSE = 1/N * (x*w -y)**2

print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')
#training
learning_rate = 0.1
n_iters = 200
loss = nn.MSELoss()
optimiser = torch.optim.SGD(model.parameters(),lr=learning_rate)



for epoch in range(n_iters):
    #preds = forqaed pass
    y_pred = model(X)

    #loss 
    l = loss(y,y_pred)
    #gradients = backward pass
    l.backward() #dl.dw
    
    
    #update weights
    # with torch.no_grad(): #so that this step is not counted in the gradient tracking
    #     w -= learning_rate * w.grad
    optimiser.step()
    
    
    #make gradients zero before next iteration
    # w.grad.zero_()
    optimiser.zero_grad()
    if epoch %10 == 0:
        [w,b] = model.parameters()
        print(f'epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {l:0.8f}')
print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('ex1data2.txt',header=None,  names=['Size','Bedrooms', 'Price'])

data = (data - data.mean()) / data.std() 
train_y=data.Price

train_y=train_y.values.reshape(len(train_y),1)

#train_y = np.matrix(train_y.values)#this or line 8 works the same 
train_x=data[['Size','Bedrooms']]





train_x = np.column_stack((np.ones(len(data)),train_x))

print(train_x.shape)
print(train_y.shape)



#loss function

def computeCost(X, y, theta):  
    inner = np.power(((X * theta) - y), 2)
    return np.sum(inner) / (2 * len(X))



#theta=np.random.normal(size=(3,1))
theta = np.zeros(3).reshape((3,1))

theta_normal=np.matmul(np.matmul(np.linalg.inv(np.matmul(train_x.T,train_x)),train_x.T),train_y)

#hypothesis by normal equation
h_n=np.matmul(train_x,theta_normal)

j_n = np.sum(0.5*(np.square(h_n-train_y)))

print(j_n/97)

#gradient desent

n_epock=100
alpha=0.001



for i in range(n_epock):
    h=np.matmul(train_x,theta)
    theta = theta - alpha*(np.transpose((np.matmul(np.transpose(h-train_y),train_x))))/len(train_y)
    j = np.sum(np.square(h-train_y))/(2*len(train_y))
    print("for iter ",i,"the loss is",j)



print(theta)



plt.scatter(train_x[:,1],np.matmul(train_x,theta_normal), c='g', label='Model')
plt.scatter(train_x[:,1],train_y, c='r', label='True')
plt.show()


plt.scatter(train_x[:,1],np.matmul(train_x,theta), c='g', label='Model')
plt.scatter(train_x[:,1],train_y, c='r', label='True')
plt.show()
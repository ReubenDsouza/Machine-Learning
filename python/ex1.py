import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('ex1data1.txt',header=None,  names=['Population', 'Profit'])

train_y=data.Profit

train_y=train_y.values.reshape(len(train_y),1)

#train_y = np.matrix(train_y.values)#this or line 8 works the same 
train_x=data.Population
train_x=train_x.values.reshape(len(train_x),1)



data.plot(kind='scatter', x='Population', y='Profit', figsize=(12,8))

train_x = np.column_stack((np.ones(len(data)),train_x))

print(train_x.shape)
print(train_y.shape)



#loss function

def computeCost(X, y, theta):  
    inner = np.power(((X * theta) - y), 2)
    return np.sum(inner) / (2 * len(X))



theta=np.random.normal(size=(2,1))

theta_normal=np.matmul(np.matmul(np.linalg.inv(np.matmul(train_x.T,train_x)),train_x.T),train_y)

#hypothesis by normal equation
h_n=np.matmul(train_x,theta_normal)

j_n = np.sum(0.5*(np.square(h_n-train_y)))

print(j_n/97)

#gradient desent

n_epock=100
alpha=0.01


h=np.matmul(train_x,theta)
for i in range(n_epock):
    h=np.matmul(train_x,theta)
    theta = theta - alpha*(np.transpose((np.matmul(np.transpose(h-train_y),train_x))))/len(train_y)
    j = np.sum(np.square(h-train_y))/(2*len(train_y))
    print("for iter ",i,"the loss is",j)






plt.scatter(train_x[:,1],np.matmul(train_x,theta_normal), c='g', label='Model')
plt.scatter(train_x[:,1],train_y, c='r', label='True')
plt.show()


plt.scatter(train_x[:,1],np.matmul(train_x,theta), c='g', label='Model')
plt.scatter(train_x[:,1],train_y, c='r', label='True')
plt.show()



from sklearn import linear_model  
model = linear_model.LinearRegression()  
model.fit(train_x, train_y)

x = np.array(train_x[:, 1])  
f = model.predict(train_x).flatten()

fig, ax = plt.subplots(figsize=(12,8))  
ax.plot(x, f, 'r', label='Prediction')  
ax.scatter(data.Population, data.Profit, label='Traning Data')  
ax.legend(loc=2)  
ax.set_xlabel('Population')  
ax.set_ylabel('Profit')  
ax.set_title('Predicted Profit vs. Population Size')  


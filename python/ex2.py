# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 01:54:50 2018

@author: user
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv('ex2data1.txt',header=None,names=['exam1','exam2','admitted'])

positive=data[data['admitted'].isin([1])]

negative=data[data['admitted'].isin([0])]

file,ax= plt.subplots(figsize=(12,8))  

ax.scatter(positive['exam1'], positive['exam2'], s=50, c='b', marker='o', label='Admitted')  
ax.scatter(negative['exam1'], negative['exam2'], s=50, c='r', marker='x', label='Not Admitted')  
ax.legend()  
ax.set_xlabel('Exam 1 Score')  
ax.set_ylabel('Exam 2 Score')


def sigmoid(x):
    return 1/(1+np.exp(-x))

y=data.admitted
x=data[['exam1','exam2']]


y = y.values.reshape([len(data),1])

x = np.column_stack((np.ones(len(y)),x))

#theta=np.random.normal(size=(3,1))
theta=np.array([[0],[0],[0]])




def cost(theta,x,y):
 
    theta=np.reshape(theta,(3,1))
    x=np.reshape(x,(len(x),3))

    y=np.reshape(y,(len(y),1))


    first= np.matmul(-y.T,np.log(sigmoid(np.matmul(x,theta))))
    second=np.matmul((1-y).T,np.log(1-(sigmoid((np.matmul(x,theta))))))
    return np.sum(first-second)/(len(x))


def gradient (theta,x,y):

    theta=np.reshape(theta,(3,1))

    x=np.reshape(x,(len(x),3))

    y=np.reshape(y,(len(y),1))
 
    grad=(np.matmul(x.T,(sigmoid(np.matmul(x,theta)))-y))/len(x)
    
    return grad



import scipy.optimize as opt  
result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(x, y))  
print(cost(result[0],x,y))


prediction=np.random.normal(size=(len(y),1))
def predict(x,theta):
    for i in range(len(x)):
        if sigmoid(np.dot(x[i,:],theta))>0.5:
            prediction[i]=1
        else:
            prediction[i]=0
    
theta_min = np.matrix(result[0]) 
theta_min=np.reshape(theta_min,(3,1))
predict(x,theta_min)
'''
predictions = predict(x,theta_min)  
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]  
accuracy = (sum(map(int, correct)) % len(correct))  
print('accuracy = {0}%'.format(accuracy))  
'''

from sklearn.metrics import accuracy_score

print(accuracy_score(y,prediction))
    






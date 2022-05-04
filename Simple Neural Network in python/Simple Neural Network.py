#the neural network is trying to predict what the value of y is based on x

import numpy as np

def nonlin(x,deriv=False):                    # sigmoid function

    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))


Z = np.array([[0,0,1],                          # new input dataset
              [0,1,1],
              [1,1,1],
              [1,1,0],
              [1,1,1],
              [1,1,0],
              [1,1,0],
              [1,0,0],
              [1,0,0]])


                                                # input dataset
X = np.array([[0,0,1],
              [0,1,1],
              [1,1,1],
              [1,0,1],
              [1,1,0],
              [1,0,0]])
    
          
y = np.array([[1,1,1,1,0,0]]).T                                # output dataset  

np.random.seed(1)                               # seed random numbers to make calculation

syn0 = 2*np.random.random((3,1)) - 1            # initialize weights randomly with mean 0

for iter in range(10000):

   
    l0 = X                                       # forward propagation
    l1 = nonlin(np.dot(l0,syn0))
    
    l1_error = y - l1                           # how much did it miss 
    
    l1_delta = l1_error * nonlin(l1,True)       # multiply how much we missed by the slope of the sigmoid at the values in l1
    
    syn0 += np.dot(l0.T,l1_delta)               # update weights


def new_data():
    
    l0 = Z                                       # forward propagation
    l1 = nonlin(np.dot(l0,syn0))
    print(l1, '\n')
    
    
    number = 0 
    for i in range(len(l1)):
        if l1[number] > 0.5:
            print("1")
        else:
            print("0")
        number += 1

        
print("\nStarting Data\n", X, "\n\n", y, "\n")
print("Output After Training: ")
print(l1)
print("\n")


number = 0 
for i in range(len(l1)):
    if l1[number] > 0.5:
        print("1")
    else:
        print("0")
    number += 1
    

print("\n\nStarting Data")  
print(Z, "\n")
new_data()

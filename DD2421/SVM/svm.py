import numpy as np
import random, math
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Generating Data
np.random.seed(100)

xi = np.random.randn(20, 2)
xj = np.random.randn(20, 2)

classA = xi*0.2 + [1.5, 0.5]
classB = xj*0.2 + [0, -0.5]

inputs = np.concatenate((classA, classB))
targets = np.concatenate((np.ones(classA.shape[0]), -np.ones(classB.shape[0])))

N = inputs.shape[0] # Number of rows (samples)

permute = list(range(N))
random.shuffle(permute)
inputs = inputs[permute, :]
targets = targets[permute]


# plotting

#plt.plot([p[0] for p in classA], [p[1] for p in classA], 'bo')
#plt.plot([p[0] for p in classB], [p[1] for p in classB], 'ro')
#plt.axis('equal')

#plt.show()

def linKernel(xi, xj):
    return np.dot(xi[i], xj[i])


def initP(Data):
    N= len(Data)
    P = np.zeros(N,N)
    for  i in range(N):
        for j in range(N):
            
            
                return targetKernel

P = initGlobal(targets)


#def objective(alpha):
    


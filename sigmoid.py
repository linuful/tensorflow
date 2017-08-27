import math
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    a = []
    for item in x:
        a.append(1/(1+math.exp(-item)))
    return a


def tanh(x):
    a=[]
    for item in x:
        a.append((math.exp(item) - math.exp(-item))/(math.exp(item)+math.exp(-item)))
    return a;

x = np.arange(-10., 10., 0.2)
sig = sigmoid(x)
plt.plot(x,sig)
plt.show()

x1=np.arange(-10.0,10.0,0.2)
sig=tanh(x1)
plt.plot(x1,sig)
plt.show()
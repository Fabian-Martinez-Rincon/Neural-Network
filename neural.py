import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

from sklearn.datasets import make_circles
#Creamos el dataset
n = 500
p = 2

X, Y = make_circles(n_samples=n, factor = 0.5,noise = 0.05)

plt.scatter(X[Y==0, 0], X[Y == 0,1],c="skyblue")
plt.scatter(X[Y==1, 0], X[Y == 1,1],c="salmon")
plt.axis("equal")
plt.show()

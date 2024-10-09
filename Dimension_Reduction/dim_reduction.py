import random
import numpy as np
import matplotlib.pyplot as plt

def main():
    x = np.random.multivariate_normal([0, 0], np.diag([10, 0.1]),size=100)
    #rotaiton matrix
    theta = np.pi/3
    R = np.array([[np.cos(theta), np.sin(theta)],
                  [-np.sin(theta), np.cos(theta)]])
    x = x @ R
    #plt.scatter(x[:, 0], x[:, 1], marker ='o')
    #plt.axis('equal')
    #plt.show()

    # Calculating SVD components
    U, s, VT = np.linalg.svd(x)
    projection = x @ VT[:,1]
    plt.scatter(projection, np.zeros(projection.shape), marker ='o')
    plt.show()

if __name__ == "__main__":
    main()
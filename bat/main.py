import random
from BatAlgorithm import *

def Fun(D, sol):
    val = 0.0;
    for i in range(D):
        val = val + sol[i] * sol[i];
    
    return val;

# For reproducive results
#random.seed(5)

if __name__ == "__main__":

    for i in range(1):
        Algorithm = BatAlgorithm(2, 10, 1000, 0.5, 0.5, 0.0, 2.0, -10.0, 10.0, Fun);
        Algorithm.move_bat();
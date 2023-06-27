from BatAlgorithm import *

def Fun(D, sol):
    val = 0.0;
    for i in range(D):
        val = val + sol[i] * sol[i];
    
    return val;


if __name__ == "__main__":

    for i in range(10):
        Algorithm = BatAlgorithm(2, 40, 100, 0.5, 0.5, 0.0, 2.0, -10.0, 10.0, Fun);
        Algorithm.move_bat();
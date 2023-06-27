import random 
import numpy as np
from math import inf

class BatAlgorithm():
    def __init__(self, D, M, N, A, r, freqMin, freqMax, Lower, Upper, function):
        self.D = D;  # dimensions
        self.M = M;  # population size 
        self.N = N;  # number of itterations


        self.A = A;  # loudness
        self.r = r;  # pulse rate
        self.freq = [0] * self.M;  # frequency
        self.v = [[0 for i in range(self.D)] for j in range(self.M)];  # velocity
        self.Sol = [[0 for i in range(self.D)] for j in range(self.M)];  # Current solution (coordinates)


        self.freqMin = freqMin;  # frequency min
        self.freqMax = freqMax;  # frequency max
        self.Lower = Lower;  # lower bound
        self.Upper = Upper;  # upper bound

        self.f_min = 0.0;  # minimum fitness
        
        self.Fitness = [0] * self.M;  # fitness
        self.best = [0] * self.D;  # closest bat
        self.Fun = function; # sum of squared coordinates


    def best_bat(self): # Looking for the closest bat to the pray
        val, idx = min((val, idx) for (idx, val) in enumerate(self.Fitness));
        
        for i in range(self.M): 
            if (self.Fitness[idx] < self.Fitness[i]):
                idx = i;
        
        for i in range(self.D):
            self.best[i] = self.Sol[idx][i];
        
        self.f_min = self.Fitness[idx];


    def init_bat(self): # Bat creation
        for i in range(self.M):
            self.freq[i] = 0; # Starting freq
            
            for j in range(self.D):
                rnd = np.random.uniform(0, 1);
                self.v[i][j] = 0.0; # Zeroed velocity
                self.Sol[i][j] = self.Lower + (self.Upper - self.Lower) * rnd; # Current coordinates based on equation

            self.Fitness[i] = self.Fun(self.D, self.Sol[i]); # Swarn current fitness
        
        self.best_bat();


    def bounds(self, val, lower, upper):
        if (val < lower):
            val = lower;
        
        if (val > upper):
            val = upper;

        return val;

    
    def move_bat(self):
        S = [[0.0 for i in range(self.D)] for j in range(self.M)];

        self.init_bat(); # Bat swarn creation

        for t in range(self.N): 
            for i in range(self.M):
                rnd = np.random.uniform(0, 1); # (0; 1)
                self.freq[i] = self.freqMin + (self.freqMax - self.freqMin) * rnd; # Frequency for every bat

                for j in range(self.D):
                    self.v[i][j] = self.v[i][j] + (self.Sol[i][j] - self.best[j]) * self.freq[i]; 
                    # Vecolity of every bat based on equation containing frequency 

                    S[i][j] = self.Sol[i][j] + self.v[i][j]; # Coordinates based on equation containing prev. iteration  

                    S[i][j] = self.bounds(S[i][j], self.Lower, self.Upper);  

                rnd = np.random.random_sample(); # [0; 1)

                if rnd > self.r: # Based on the pulserate we can determinate if there is a pray  
                    for j in range(self.D): 
                        S[i][j] = self.best[j] + 0.001 * random.gauss(0, 1);     
                        S[i][j] = self.bounds(S[i][j], self.Lower, self.Upper);
                        
                Fnew = self.Fun(self.D, S[i]); # New Fitness

                rnd = np.random.random_sample(); # [0; 1)

                if (Fnew <= self.Fitness[i]) and (rnd < self.A): # Check if the answer is better &
                                                                 # Loudnesss is lower, than we change our:
                                                                 # Current coordinates and fitness
                    for j in range(self.D):
                        self.Sol[i][j] = S[i][j];
                    self.Fitness[i] = Fnew; 

                if (Fnew <= self.f_min): # Overall optimal fitness check
                    for j in range(self.D): 
                        self.best[j] = S[i][j]; 
                    self.f_min = Fnew;

        print(self.f_min);
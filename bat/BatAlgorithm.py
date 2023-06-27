import random 
import numpy as np

class BatAlgorithm():
    def __init__(self, D, NP, N_Gen, A, r, Qmin, Qmax, Lower, Upper, function):
        self.D = D  # dimension
        self.NP = NP  # population size 
        self.N_Gen = N_Gen  # generations
        self.A = A  # loudness
        self.r = r  # pulse rate
        self.Qmin = Qmin  # frequency min
        self.Qmax = Qmax  # frequency max
        self.Lower = Lower  # lower bound
        self.Upper = Upper  # upper bound

        self.f_min = 0.0  # minimum fitness
        
        self.Lb = [0] * self.D  # lower bound
        self.Ub = [0] * self.D  # upper bound
        self.Q = [0] * self.NP  # frequency

        self.v = [[0 for i in range(self.D)] for j in range(self.NP)]  # velocity
        self.Sol = [[0 for i in range(self.D)] for j in range(self.NP)]  # population of solutions
        self.Fitness = [0] * self.NP  # fitness
        self.best = [0] * self.D  # best solution
        self.Fun = function


    def best_bat(self):
        i = 0
        j = 0
        for i in range(self.NP):
            if self.Fitness[i] < self.Fitness[j]:
                j = i
        for i in range(self.D):
            self.best[i] = self.Sol[j][i]
        self.f_min = self.Fitness[j]

    def init_bat(self):
        for i in range(self.D):
            self.Lb[i] = self.Lower
            self.Ub[i] = self.Upper

        for i in range(self.NP):
            self.Q[i] = 0 # Частота
            for j in range(self.D):
                rnd = np.random.uniform(0, 1)
                self.v[i][j] = 0.0 # Швидкість
                self.Sol[i][j] = self.Lb[j] + (self.Ub[j] - self.Lb[j]) * rnd # Стартова рандомна координата
            self.Fitness[i] = self.Fun(self.D, self.Sol[i]) # Хуйня для знаходження x_best
        
        self.best_bat() # Знаходження x_best(яка з мишей найблища до здобичі)

    def simplebounds(self, val, lower, upper):
        if val < lower:
            val = lower
        if val > upper:
            val = upper
        return val

    def move_bat(self):
        S = [[0.0 for i in range(self.D)] for j in range(self.NP)]

        self.init_bat() # Нова миш

        for t in range(self.N_Gen):
            for i in range(self.NP):
                rnd = np.random.uniform(0, 1) # (0; 1)
                self.Q[i] = self.Qmin + (self.Qmax - self.Qmin) * rnd # Частота для кожної миші 

                for j in range(self.D):
                    self.v[i][j] = self.v[i][j] + (self.Sol[i][j] - self.best[j]) * self.Q[i] # Швидкість миші на данній ітерації
                    # self.best - x_best координата мині яка найблище до здобичі
                    # self.Sol[i] - координата даної миші 

                    S[i][j] = self.Sol[i][j] + self.v[i][j] # Координата миші залежно від координати на попередній ітерації

                    S[i][j] = self.simplebounds(S[i][j], self.Lb[j], self.Ub[j]) # Перевірка на грані  

                rnd = np.random.random_sample() # [0; 1)

                if rnd > self.r: # По пульсації визначаємо чи була знайдена здобич(якось змінюється координата)
                    for j in range(self.D):
                        S[i][j] = self.best[j] + 0.001 * random.gauss(0, 1)     
                        S[i][j] = self.simplebounds(S[i][j], self.Lb[j],
                                                self.Ub[j])
                        
                Fnew = self.Fun(self.D, S[i]) # Сума квадратів координат 

                rnd = np.random.random_sample() # [0; 1)

                if (Fnew <= self.Fitness[i]) and (rnd < self.A): # Якщо фітнес(сумарна відстань по всім координатам) зменшилась
                                                                 # і гучність зменшилась то змінити координати на данній ітерації
                    for j in range(self.D):
                        self.Sol[i][j] = S[i][j]
                    self.Fitness[i] = Fnew # Новий фітнес

                if Fnew <= self.f_min: # Якщо новий фітнес краще ніж мінімальний з всіх то ставимо
                    for j in range(self.D): 
                        self.best[j] = S[i][j] # Якщо фітнес краще значить і x_best зміниться 
                    self.f_min = Fnew

        print(self.f_min)
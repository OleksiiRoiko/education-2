import random

def NN_TSP():
    f = open("example19.txt",'r')
    arr = []
    while True:
        a = f.readline().split()
        if not a:
            break   
        position_1, position_2, cesta = a
        position_1, position_2, cesta = int(position_1), int(position_2), int(cesta)
        arr.append([position_1,position_2,cesta])
    f.close()

    path = [1] 
    all = [i for i in range(0,position_2)] 
    cost = 0
    a = 0
     
    while len(all) != len(path): 
        cost += find(arr,path) 
    
    for i in range(len(path) - 1): 
        for e in arr:  
            if e[0]==path[0] and e[1] == path[-1]: 
                a = e[2] 

    my_file = open("solution.txt", 'w')
    my_file.write(str(cost + a))
    my_file.write('\n')
    my_file.write(str(path))
    my_file.close()

def find(arr,path):
    last = path[-1]
    m = 1000
    b = -1
    p = 0
    for i in arr:
        if i[0] == last and i[1] not in path and i[2] < m:
            m = i[2]
            b = i[1]
            p = i[2]
        elif i[1] == last and i[0] not in path and i[2] < m:
            m = i[2]
            b = i[0]
            p = i[2]
            
    path.append(b)
    return p

NN_TSP()
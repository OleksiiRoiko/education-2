# from itertools import permutations 

# perm = permutations([1,2,3,4,5,6,7,8,9], 7)
# perm1 = permutations(range(10), 4) # send
# perm2 = permutations([1,2,3,4,5,6,7,8,9], 3) # more
#                                    #money
#                                    # i[0] - S  i[2] - N   j[1] - O
#                                    # i[1] - E  i[3] - D   j[2] - R  j[3]
# for i in perm1:
#     for j in perm2:
#         #     S              E             N           D           O        R            E         O             N             E         (D  +  E)Y
#         if (((i[0] * 1000 + i[1] * 100 + i[2] * 10 + i[3]) + (j[1] * 100 + j[2] * 10 + i[1])) == (j[1] * 1000 + i[2] * 100 + i[1] * 10 + (i[3] +i[1]))):
#             print(i[0] * 1000 + i[1] * 100 + i[2] * 10 + i[3],j)
#             print(0 + j[1] * 100 + j[2] * 10 + i[1])


from itertools import permutations 

perm = permutations([0,1,2,3,4,5,6,7,8,9], 8)


for i in perm:
#         S            E         N         D
    if((i[0]*1000  + i[1]*100 + i[2]*10 + i[3]) + (i[4] * 1000 + 100 * i[5] + 10 * i[6] + i[1]) == (10000 * i[4] + 1000*i[5]+i[2]*100+i[1]*10+i[7])):
        print(i[0]*1000  + i[1]*100 + i[2]*10 + i[3])
        if i[4] * 1000 + 100 * i[5] + 10 * i[6] + i[1] < 1000:
            print('0'+ str(i[4] * 1000 + 100 * i[5] + 10 * i[6] + i[1]))
        else:
            print(i[4] * 1000 + 100 * i[5] + 10 * i[6] + i[1])
        
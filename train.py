#encoding=utf-8
from __future__ import print_function, division # python2 -> 3
import numpy as np 
import copy # for deep copy arrays
from load import LoadData

# area, perimeter, compactness, length, width, asymmetry coefficient, and length of kernel groove
# The three types of wheat (classes) are: 1 = Kama, 2 = Rosa and 3 = Canadian
# TrainSeeds.csv contains 55 data points / class
# TestSeeds.csv contains 15 data points / class




trainsetpath = "./trainSeeds.csv"
testsetpath = "./testSeeds.csv"
trainmat, traintarget = LoadData(trainsetpath)
testmat, testtarget = LoadData(testsetpath)

c = 0.001 # learning rate
class_n = int(np.ptp(traintarget)+1)
# print(class_n) # =3
W = np.zeros((class_n, trainmat.shape[1])) # initialize weights
# print(W.shape) #(3,8)

for k in range(5):
    for i in range(trainmat.shape[0]):
        x = trainmat[i:i+1].transpose() # dim stays
        y = np.zeros((class_n,1))
        # print(traintarget[i,0]-1)
        # print(traintarget[i])
        y[int(traintarget[i])-1,0] = 1
        # print(x.shape) #x is a vector with shape (9,1)
        out = W.dot(x) # activation function, no non-linearity
        # print(out)
        dW = c*x.transpose() 
        sign = True
        for j in range(class_n):
            if y[j,0] > out[j,0]:
                # print(W[j:j+1].shape)
                # print(dW.shape)
                W[j:j+1] += dW
                sign = True
            elif y[j,0] < out[j,0]:
                W[j:j+1] -= dW
                sign = False
        if i == 0:
            print(out,y, sep="\n")
            print("dw = ", end="")
            print(dW)
            print(sign)
        if i == 64:
            break
        
    # print(W)
    

    
correct_cnt = 0.0
total_cnt = testmat.shape[0]
for i in range(testmat.shape[0]):
    x = testmat[i:i+1].transpose()
    y = np.zeros((class_n,1))
    y[int(testtarget[i])-1,0] = 1
    out = W.dot(x)
    # print(out)
    if np.argmax(out) == np.argmax(y):
        correct_cnt += 1
print(correct_cnt)
print(total_cnt)
print("accuracy = {}".format(correct_cnt/total_cnt))




















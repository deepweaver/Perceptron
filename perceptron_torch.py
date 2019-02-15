import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from load import LoadData 
# print(torch.__version__)

trainsetpath = "./trainSeeds.csv"
testsetpath = "./testSeeds.csv"
class_n = 3
attr_n = 7
epochs = 1000
num_epochs = 100
learning_rate = 0.00001
trainmat, traintarget = LoadData(trainsetpath)
traininput =torch.from_numpy( trainmat[:,:-1])
testmat, testtarget = LoadData(testsetpath)
testinput = torch.from_numpy(testmat[:,:-1])

def to_onehot(target):
    # print(target.shape)
    onehot = -np.ones((target.shape[0],class_n),dtype=np.float32)
    for i in range(target.shape[0]):
        onehot[i,int(target[i]-1)] = 1
    return onehot
onehottraintarget = torch.from_numpy(to_onehot(traintarget))
onehottesttarget = torch.from_numpy(to_onehot(testtarget))

perceptron = nn.Linear(attr_n, class_n, bias=True)
# criterion = nn.L1Loss()
def criterion(out, label):
    return torch.sum(torch.sum(label - out))**2
# criterion = nn.MSELoss()
optimizer = torch.optim.SGD(perceptron.parameters(), lr=learning_rate)
# print(traininput.shape)
perceptron(traininput)

for epoch in range(epochs):
    trainoutput = perceptron(traininput)
    # print(trainoutput.shape)
    # print(onehottraintarget.shape)
    loss = criterion(trainoutput, onehottraintarget)
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch+1) % 100 == 0:
        print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
print(onehottraintarget)
print(trainoutput)
# to_onehot(traintarget)
perceptron.eval()
predicted = perceptron(testinput)
predicted = predicted.detach().numpy()
correct_cnt = 0
predicted = np.argmax(predicted, axis=1)
# # print(np.argmax(testtarget, axis=1))
# print(predicted.shape)
# print(testtarget==predicted)
for i in range(45):
    if testtarget[i] == predicted[i]:
        correct_cnt += 1
print(correct_cnt)
print("accuracy = {}/45 = {}".format(correct_cnt, correct_cnt/45))








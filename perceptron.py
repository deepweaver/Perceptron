import numpy as np
from load import LoadData
class Perceptron(object):

    def __init__(self, no_of_inputs, threshold=1000, learning_rate=0.1, epsilon = 0.0001):
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.weights = np.zeros(no_of_inputs + 1)
        self.epsilon = epsilon
           
    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        return 1 if summation > 0 else 0

    def train(self, training_inputs, labels):
        for _ in range(self.threshold):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)

# helper function to convert label of different classes to one-hot vector
def createlabel(traintarget, typecode):
    labels = np.zeros(traintarget.shape)
    for i in range(traintarget.shape[0]):
        if traintarget[i] == typecode:
            labels[i] = 1
    return labels 

if __name__ == "__main__":
    trainsetpath = "./trainSeeds.csv"
    testsetpath = "./testSeeds.csv"
    trainmat, traintarget = LoadData(trainsetpath)
    testmat, testtarget = LoadData(testsetpath)

    training_inputs = []
    for i in range(trainmat.shape[0]):
        training_inputs.append(trainmat[i,:-1])

    initfinalweights = ''
    # training ...
    # use three different perceptrons, three sets of weights to predict categories
    initfinalweights += "These are initial weights:\n"
    perceptron1 = Perceptron(trainmat.shape[1]-1)
    initfinalweights += str(perceptron1.weights)
    perceptron1.train(training_inputs, createlabel(traintarget, 1))
    perceptron2 = Perceptron(trainmat.shape[1]-1)
    initfinalweights += str(perceptron2.weights)
    perceptron2.train(training_inputs, createlabel(traintarget, 2))
    perceptron3 = Perceptron(trainmat.shape[1]-1)
    initfinalweights += str(perceptron3.weights)
    perceptron3.train(training_inputs, createlabel(traintarget, 3))
    initfinalweights += "\nThese are final weights(after training):\n"
    initfinalweights += str(perceptron1.weights)
    initfinalweights += str(perceptron2.weights)
    initfinalweights += str(perceptron3.weights)
    initfinalweights += "\n\n"
    
    
    
    # testing ...
    testing_inputs = []
    for i in range(testmat.shape[0]):
        testing_inputs.append(testmat[i,:-1])
    targetlabels1 = createlabel(testtarget,1)
    targetlabels2 = createlabel(testtarget,2)
    targetlabels3 = createlabel(testtarget,3)  
    correct_cnt = 0.0
    total_cnt = testtarget.shape[0]
    OrigAndPred = "Original and predicted class values:\n" # write to file log.txt
    output3 = np.zeros((3))
    for i in range(testtarget.shape[0]):
        output3[0] = perceptron1.predict(testing_inputs[i])
        output3[1] = perceptron2.predict(testing_inputs[i])
        output3[2] = perceptron3.predict(testing_inputs[i])
        # create one-hot prediction vector
        if  output3[0] == targetlabels1[i] == 1:
            OrigAndPred += '{}\t1\n'.format(str(int(testtarget[i])))
            correct_cnt += 1 
        elif output3[1] == targetlabels2[i] == 1:
            OrigAndPred += '{}\t2\n'.format(str(int(testtarget[i])))
            correct_cnt += 1
        elif output3[2] == targetlabels3[i] == 1:
            OrigAndPred += '{}\t3\n'.format(str(int(testtarget[i])))
            correct_cnt += 1
        else:
            tmp = "None" if np.sum(output3) == 0 else int(np.argmax(output3)+1)
            OrigAndPred += '{}\t{}\twrong prediction\n'.format(str(int(testtarget[i])), str(tmp))
    # if the first perceptron don't recognize it, then ask the second ...
    print(correct_cnt)
    print(total_cnt)
    print("accuracy = {}".format(correct_cnt/total_cnt))
    # print(OrigAndPred)


    with open("./log.txt", 'w+') as file:
        file.write(initfinalweights)
        file.write(OrigAndPred)
        file.write("Total number of iteration is 1000 and the \nterminating criteria is 0.001") # it's the threshold value and epsilon
    
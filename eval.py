import numpy as np
import utils2
from vals import val
'''evaluate dataset at a certain epoch weight'''

conv1 = utils2.convolution(5,5)
conv2 = utils2.convolution(5,5)


fc1 = utils2.fully_connected(16, 10)

with open(r"C:\Users\johna\cnn\epochweights3.txt", "r") as file:
    ct = 0 
    for line in file:
        l = line.split(" ")[0:-1]
        l = np.array([val(float(i),mod=False) for i in l])
        if ct == 0 :
            ct+=1
            conv1.conv = l.reshape((5,5))
        elif ct == 1:
            ct+=1
            conv2.conv = l.reshape((5,5))
        elif ct == 2:
            fc1.weight = l.reshape((16,10))
            ct+=1
        else:
            fc1.bias = l

TP = [0] * 10 
FP = [0] * 10 
FN = [0] * 10 
ct = 0
with open(r"C:\Users\johna\cnn\finished\data_set\test\testcopy.txt", "r") as file:
    for line in file:
        datapt = line.split(" ")
        fval = int(datapt[0])
        datapt = datapt[1:-1]
        datapt = np.array([val(int(i)) for i in datapt]).reshape((1,28,28))/255
        ct+=1
        used = []
        conv1out = conv1.forward(datapt,used) # pass throug model
        relu1 = utils2.relu(conv1out)
        max1out = utils2.maxpool(relu1,(2,2))
        conv2out = conv2.forward(max1out,used)
        relu2 = utils2.relu(conv2out)
        max2out = utils2.maxpool(relu2,(2,2)).flatten()
        fcout = fc1.forward(max2out,used)
        sft_max = utils2.softmax(fcout)
        beeg = np.argmax(sft_max)
        print(sft_max)
        print(fval)
        if beeg == fval:
            TP[fval]+=1
        else:
            FN[fval]+=1
            FP[beeg]+=1
        
        print(ct)
        if ct == 400:
            break

print(TP) # true positives

# print(FP)
# print(FN)
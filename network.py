from vals import val
import utils2
import numpy as np
from dataloader import data
from PIL import Image
import sys

filename = r"C:\Users\johna\cnn\finished\data_set\train\train.txt"
mnist_data = data(filename)
mnist_data.load()
# load mnist data

conv1 = utils2.convolution(5,5)
conv2 = utils2.convolution(5,5)

fc1 = utils2.fully_connected(16, 10) # fc and conv layers
ct = 1
out_temp = "epoch"
f2 = open("epochres.txt", "w") # output loss


f_template = "epochweights"
for epoch in range(50): # perform 50 epochs, ended early 
    data_lab, dataset = mnist_data.sample(10) # sample the dataset tin sies of 10
    used = []
    tot_loss = 0 
    for i in range(len(dataset)): # go across every sample
        for j in range(len(dataset[i])): # for every item in the sample
            datapt = dataset[i][j]
            used = []
            conv1out = conv1.forward(datapt,used)
            relu1 = utils2.relu(conv1out)
            max1out = utils2.maxpool(relu1,(2,2))
            conv2out = conv2.forward(max1out,used)
            relu2 = utils2.relu(conv2out)
            max2out = utils2.maxpool(relu2,(2,2)).flatten()
            fcout = fc1.forward(max2out,used)
            sft_max = utils2.softmax(fcout)
            loss = utils2.cross_entropy_loss(sft_max, data_lab[i][j]) 
            lbl =  data_lab[i][j]
            
            tot_loss +=loss.data
            if loss.data <0:
                print(fcout)
                print('NEGATIVE LOSS')
                sys.exit()
            if i != len(dataset)-1: # accumulate the gradients
                loss.backward()
            
        loss.backward(update = "adam",timestep = ct) # perform the backward and update the values
        loss.zero_grad()
        tot_loss +=loss.data
        print("Iter: ", str(ct) , " Loss: ",str(loss))
        if loss.data <0:
            print(fcout)
        print(sft_max)
        print(lbl)
        ct+=1
    f = f_template + str(epoch) + '.txt'
    file = open(f, "w") # write into files
    for i in conv1.conv.flatten():
        file.write(str(i.data) + " ")
    file.write('\n')
    for i in conv2.conv.flatten():
        file.write(str(i.data) + " ")
    file.write('\n')
    for i in fc1.weight.flatten():
        file.write(str(i.data) + " ")
    file.write('\n')

    for i in fc1.bias.flatten():
        file.write(str(i.data) + " ")
    
    file.close()
    print("Epoch: "+ str(epoch) , " Loss: ",str(tot_loss/2000))
    f2.write("Epoch: "+ str(epoch) + " Loss: "+str(tot_loss/2000)+'\n')

f2.close()
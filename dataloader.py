import numpy as np
import random 
from vals import val

class data:
    ''' dataloader'''

    def __init__(self, filename):
        self.labels = []
        self.values = []
        self.filename = filename

    def load(self, sz = 500):
        '''
        load the data every line
        '''
        with open(self.filename) as file:
            ct = 0 
            for line in file:
                s = line.rstrip()
                s = s.split(" ")
                label = np.zeros(10)
                label[int(s[0])] = 1
                label = [val(i,mod=False) for i in label]
                self.labels.append(label)
                s = [val(int(i),mod=False) for i in s[1:]]
                s = np.array(s)/255
                s = s.reshape((1,28,28))
                self.values.append(s)
                ct+=1
                if ct%100 == 0 :
                    print(ct)
            print("done loading")
        self.labels = np.array(self.labels)
        self.values = np.array(self.values)
    def smpl(self,size,sz):
        '''
        sample the dataset with a max size of sz and samples of size
        '''
        n = len(self.values)
        mixed = random.sample(range(n),sz)
        temp_label = self.labels[mixed]
        temp_vals = self.values[mixed]
        sample_label = [[] for i in range(int(np.ceil(sz/size)))]
        sample_vals = [[] for i in range(int(np.ceil(sz/size)))]
        
        for i in range(sz):
              
            sample_label[i//size].append(temp_label[i])
            sample_vals[i//size].append(temp_vals[i])

        return sample_label, sample_vals
    def sample(self,size):
        '''
        samples the dataset with samples of size, really no reason for this because we're not batching because we aren't using CUDA
        '''
        n = len(self.values)
        mixed = random.sample(range(n),n) # mix indices and rearrange data acccording to mixed indices this is necessary because everything is currently in groups by label
        temp_label = self.labels[mixed]
        temp_vals = self.values[mixed]
        sample_label = [[] for i in range(int(np.ceil(n/size)))]
        sample_vals = [[] for i in range(int(np.ceil(n/size)))]
        
        for i in range(n):
              
            sample_label[i//size].append(temp_label[i])
            sample_vals[i//size].append(temp_vals[i])

        return sample_label, sample_vals
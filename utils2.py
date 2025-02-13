import numpy as np
from vals import val
# from collections import heapq
from collections import deque
class fully_connected: # fully connected layer
    def __init__(self, value_size, output_size):
        randbias = 2*np.random.rand(output_size)-1
        self.bias = np.array([val(i) for i in randbias]) # random bias of size output

        randweight = 2*np.random.rand(value_size,output_size)-1
        randweight = randweight.flatten()
        randweight = np.array([val(i) for i in randweight])
        self.weight =  np.reshape(randweight, (value_size,output_size)) # random weights of size valuexoutput
        
    def forward(self,value, used):
        used.append(self) # append it to used
        return value @ self.weight +self.bias # pass through fc layer
    

class convolution:
    def __init__(self, x,y):
        randconv = 2*np.random.rand(x,y)-1
        randconv = randconv.flatten()
        randconv = np.array([val(i) for i in randconv])
        self.conv =  np.reshape(randconv, (x,y)) # random convolution window size

    def forward(self,value,used,stride =1):
       
    #    if len(value.shape) == 3: #hard code rgbimage
        h = (value.shape[1] - self.conv.shape[0]) // stride + 1
        w = (value.shape[2] - self.conv.shape[1]) // stride + 1
        res = np.zeros(shape=(value.shape[0], h,w))
        res = res.flatten()
        res = np.array([val(i) for i in res])
        res = np.reshape(res, (value.shape[0], h,w))
        # print(res)
        for l in range(0,value.shape[0]):
            for i in range(0, w):
                for j in range(0, h):
                    region = value[l,i * stride : i * stride + self.conv.shape[0], j * stride : j * stride + self.conv.shape[1]]
                    res[l,i, j] = np.sum(region * self.conv) # pass through the convolution window and perfomr the convolution
        used.append(self)
        return res
       


def cross_entropy_loss(guess, truth):

    logs =  np.vectorize(lambda x: x.ln())(guess) # cross entropy loss formula
    loss = truth@logs
    return  -loss

def maxpool(inp, shape):
    x_shape = shape[0]
    y_shape = shape[1]
    res = []
    
    for i in range(inp.shape[0]):
        for j in range(0, inp.shape[1], y_shape):
            for k in range(0, inp.shape[2],x_shape):
                arr_int = inp[i, j:j+y_shape, k:k+x_shape].flatten()
                max_val = arr_int[np.argmax(arr_int)]
                new_val = val(0,mod = False) + max_val
  

                res.append(new_val)

    res = np.array(res)
    
    f_shape = (inp.shape[0], int(np.ceil(inp.shape[1]/y_shape)), int(np.ceil(inp.shape[2]/x_shape)))
    res = np.reshape(res,f_shape)

    return res

    # really slow hardcoded 

def relu(value): # perform the relu on every value in the np
    return np.vectorize(lambda x: x.relu())(value)

def softmax(value): # perform a softmax
    exp = np.vectorize(lambda x: x.exp())(value)
    # k = np.sum(exp)
    tot = np.sum(exp)

    res = exp/tot
    

    return res

def khan(node): # khan's redefined here
    res = []
    stack = [node]
    inedge = {}
    graph = {}
    while(len(stack)!=0):
        s = stack.pop()
        inedge[s] = len(s.prev)
        if graph.get(s) is None:
            graph[s] = []
        for k in s.prev:
            if inedge.get(k) is None:
                stack.append(k)
            if graph.get(k) is None:
                graph[k] = []
            graph[k].append(s)
    
    queue = deque()
    for key,values in inedge.items():
        if values == 0 :
            queue.append(key)
    while(len(queue)!=0):
        nod = queue.popleft()
        res.append(nod)
        for l in graph[nod]:
            inedge[l] -=1
            if inedge[l] == 0:
                queue.append(l)

    return res
import numpy as np
from collections import deque


class val:
    '''
    Value class, 
    _backward is just a function which corresponds to the partial derivative of the value with it's child, multiplied by the child's grad
    '''

    def __init__(self, data, child =[], mod = True):
        self.data = data # value inside
        self._backward = lambda:None #empty function
        self.prev = set(child)
        self.grad = 0 #gradient
        self.mod = mod #mod meaning if we want to modify this value
        self.typ = ""
        self.mean = 0 # mean and var tracked for  adams
        self.var = 0
        
    
    def __add__(self, value): # addition
        if type(value)!= type(self):
            inp = val(value)
        else:
            inp = value
        res = val(inp.data+self.data, [inp, self]) # create a new res node
        def _backward(): #backprop grad is just the prev grad
            self.grad += res.grad
            inp.grad += res.grad
        
        res._backward = _backward
        res.typ = "+"
        return res
    
    def __mul__(self, value):
        if type(value)!= type(self):
            inp = val(value)
        else:
            inp = value
        res = val(inp.data*self.data, [inp, self])
        def _backward():
            self.grad += inp.data * res.grad
            inp.grad += self.data * res.grad
        res._backward = _backward
        res.typ = "*"
        return res
    
    def relu(self):
        res = val((self.data > 0) * self.data,self.prev)

        def _backward():
            self.grad = (self.data>0) * res.grad
        res._backward= _backward
        res.typ = "relu"
        return res
    
    def exp(self):
        res = val(np.exp(self.data), [self])
        def _backward():
            self.grad += (res.data) * res.grad

        res._backward = _backward
        res.typ = "exp"
        return res

    def __sub__(self, other): # self - other
        return self + (-other)

    def __neg__(self):
        res= -1*self
        res.typ = "neg"
        return res

    def __pow__(self, value):

        res = val(self.data ** value, [self])
        def _backward():
            self.grad += (value * self.data**(value-1)) * res.grad
        res._backward = _backward
        res.typ = "pow"
        return res

    def __truediv__(self, inp): 
        return self * inp**-1

    def __rtruediv__(self, inp): 
        return inp * self**-1

    def __repr__(self):
        return f'val(data = {self.data}, grad = {self.grad})'
    
    def __eq__(self, inp):
        return self.data == inp.data
    
    def __lt__(self, inp):
        return self.data< inp.data
    
    def __gt__(self, inp):
        return self.data > inp.data
    def __radd__(self, other): # other + self
        return self + other
    def __rsub__(self, other): # other - self
        return other + (-self)
    def __rmul__(self, other): # other * self
        return self * other

    def set(self,value):
        self.data = value

    def khan(self):
        #khan for toposort
        res = []
        stack = [self]
        inedge = {}
        graph = {}
        while(len(stack)!=0): # get all the nodes by dfs
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
        
        queue = deque() # queue
        for key,values in inedge.items():
            if values == 0 : # queue is queue of edges with 0 length
                queue.append(key)
        while(len(queue)!=0): # while the queue is not empty remove the first-in node and it's edges and check if children node now have 0 in-edge
            nod = queue.popleft()
            res.append(nod)
            for l in graph[nod]:
                inedge[l] -=1
                if inedge[l] == 0:
                    queue.append(l)

        return res


    def backward(self,update = "n", lr = 0.01, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8,timestep = 1):
        visited = {self}
        topo = self.khan() # technically its a backward topo
        
        self.grad = 1
        # print('a')
        for node in reversed(topo):

            node._backward() # traverse the toposort 
            if update == "gd" and node.mod is True:

                node.set(node.data - lr*node.grad) # grad desc
            if update == "adam" and node.mod is True: # adam optimizer
                node.mean = node.mean * beta1 + (1-beta1) * node.grad
                node.var = node.var * beta2 + (1-beta2) * node.grad * node.grad

                mhat = node.mean /(1-  beta1**timestep)
                vhat = node.var/(1-beta2**timestep)

                node.set(node.data - lr * mhat / np.sqrt(vhat + epsilon))

        
    def zero_grad(self):
        visited = set()
        temp = self
        visited.add(temp)
        q = [self]
        while(len(q)!=0): #dfs and zero out all the gradients
            s = q.pop() 
            s.grad =0
            for child in s.prev:
                if not child in visited:
                    visited.add(child)
                    q.append(child)
                    
    

    def ln(self):
        if self.data == 0:
            res = val(-10000, [self])
        else:
            res = val(np.log(self.data), [self])
        
        def _backward():
            if self.data == 0 :
                self.grad = 10000000 * res.grad
            else:
                self.grad += (1 / self.data) * res.grad

        res._backward = _backward
        res.typ = "ln"
        return res

    def __hash__(self): # hash code for the set is just the mem id because class definition doesnt come with hashing
        return int(bin(id(self)),2)


    
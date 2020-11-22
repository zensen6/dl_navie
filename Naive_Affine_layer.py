import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from collections import OrderedDict


def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_grad(x):
    return (1.0 - sigmoid(x))*sigmoid(x)

def softmax(x):
    m = np.max(x)
    return np.exp(x-m)/np.sum(np.exp(x-m))

def cross_entropy_loss(x,t):
    if x.ndim == 1:    # x = (n,1) 꼴이면 밑에서 계산이 batch=n이라 방지용
        t = t.reshape(1,t.size)
        x = x.reshape(1,x.size)
    batch = x.shape[0]
    delta = 1e-7
    return -np.sum(t*np.log(x+delta)) / batch

class relu():
    def __init__(self):
        pass
    def forward(self,x):
        c = x.copy()
        mask = (x<=0)
        c[mask] = 0
        return c
    
    def backward(self, L):
        c = L.copy()
        c[L>0] = 1
        c[L<=0] = 0
        return c

def numerical_gradient(func, W):
    h = 1e-4
    l = W.size
    for idx in range(0,l):
        t = W[idx]
        W[idx] = t + h
        f1 = func(W)
        W[idx] = t - h
        f2 = func(W)
        W[idx] = (f1 - f2)/(2*h)
    return W


class Affine():
    def __init__(self, w, b):
        self.w = w
        self.b = b
        self.x = None
        self.dw = None
        self.db = None
    
    def forward(self, x):
        self.x = x
        return np.dot(x,self.w) + self.b
    
    def backward(self, L):
        self.dw = np.dot(self.x.T, L)
        self.db = np.sum(L, axis = 0)
        return np.dot(L, self.w.T)

class SoftmaxwithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None
        
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_loss(self.y,t)
        return self.loss
    
    def backward(self,L=1):
        batch_size = self.t.shape[0]
        return (self.y - self.t)/batch_size
        

class TwoLayerNet:
    
    def __init__(self, input_size, hidden_size, output_size, weight_init = 0.1):
        
        self.params = {}
        self.params['w1'] = weight_init * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['w2'] = weight_init * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        
    
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['w1'], self.params['b1'])
        self.layers['Relu1'] = relu()
        self.layers['Affine2'] = Affine(self.params['w2'], self.params['b2'])
        
        self.lastLayer = SoftmaxwithLoss()
        
        
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        
        return x
    
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y,t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : 
            t = np.argmax(t, axis = 1)
        return float(np.sum(y == t))/x.shape[0]    
    
        
    def numerical_gradient(self, x, t):
        loss_func = lambda W : self.loss(x,t)
        grads = {}
        grads['w1'] = numerical_gradient(loss_func, self.params['w1']) # self.params['w1'] 가 변하면, predict값도 변해서
        # loss(x,t) 가 self.params[w1] 과 상관없어보이지만, 상관있다.
        grads['b1'] = numerical_gradient(loss_func, self.params['b1']) 
        grads['w2'] = numerical_gradient(loss_func, self.params['w2'])
        grads['b2'] = numerical_gradient(loss_func, self.params['b2']) 
        return grads
    
    
    def gradient(self, x, t):
        L = self.loss(x,t)
        dout = 1
        dout = self.lastLayer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        
        for layer in layers:
            dout = layer.backward(dout)
            
        grads = {}
        grads['w1'] = self.layers['Affine1'].dw
        grads['b1'] = self.layers['Affine1'].db
        grads['w2'] = self.layers['Affine2'].dw
        grads['b2'] = self.layers['Affine2'].db
        
        return grads
    

(x_train, t_train), (x_test, t_test) = load_mnist(normalize = True, one_hot_label = True)

N = TwoLayerNet(input_size=x_train.shape[1], hidden_size=50, output_size=10, weight_init = 0.01)

iter_nums = 10000
train_size = x_train.shape[0] # 10000
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size/batch_size, 1)

for i in range(iter_nums):
    
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    grad = N.gradient(x_batch, t_batch)
    
    for k in ['w1','b1','w2','b2']:
        N.params[k] -= learning_rate * grad[k]
    
    L = N.loss(x_batch, t_batch)
    train_loss_list.append(L)
    
    if i % iter_per_epoch == 0:
        train_accuracy = N.accuracy(x_batch, t_batch)
        train_acc_list.append(train_accuracy)
        test_accuracy = N.accuracy(x_test, t_test)
        test_acc_list.append(test_accuracy)
        
        print(train_accuracy, test_accuracy)
        

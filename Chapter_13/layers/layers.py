import numpy as np

class Layers(object):
    def __init__(self):
        self.parameters=list()

    def get_parameters(self):
        return self.parameters

class LinearLayers(Layers):
    def __init__(self,n_inputs,n_outputs):
        super.__init__()
    W = np.random.randn(n_inputs,n_outputs)*np.sqrt(2.0/n_inputs)
    self.weights = Tensor(W,autograd=True)
    self.bias = Tensor(np.zeros(n_outputs),autograd=True)

    self.parameters.append(self.weights)
    self.parameters.append(self.bias)
    def forward(self,input):
        return input.mm(self.weights)+self.bias.expand(0,len(input.data))


import numpy as np
from Chapter_13.tensor import Tensor

class Layer(object):
    def __init__(self):
        self.parameters = list()

    def get_parameters(self):
        return self.parameters
class Linear(Layer):
    def __init__(self,n_inputs,n_outputs,bias=True):
        super().__init__()
        self.use_bias = bias
        W = np.random.randn(n_inputs,n_outputs)*np.sqrt(2.0/(n_inputs))
        self.weights = Tensor(W,autograd=True)
        if self.use_bias:
            self.bias = Tensor(np.zeros(n_outputs),autograd=True)

        self.parameters.append(self.weights)
        if self.use_bias:
            self.parameters.append(self.bias)
    def forward(self,input):
        if self.use_bias:
            return input.mm(self.weights)+self.bias.expand(0,len(input.data))
        return input.mm(self.weights)

class Sequential(Layer):
    def __init__(self,layers=list()):
        super().__init__()

        self.layers = layers

    def add(self,layer):
        self.layers.append(layer)
    def forward(self,input):
        for layer in self.layers:
            input = layer.forward(input)
        return input
    def get_parameters(self):
        parms = list()
        for l in self.layers:
            parms += l.get_parameters()
        return parms

class Tanh(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.tanh()

class Sigmoid(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.sigmoid()
class Embedding(Layer):
    def __init__(self,vocab_size,dim):
        super().__init__()

        self.vocab_size = vocab_size
        self.dim = dim

        weight = np.random.rand(vocab_size,dim)-0.5 / dim
        self.weight = Tensor(weight,autograd=True)

    def forward(self,input):
        return self.weight.index_select(input)
class RNNCell(Layer):

    def __init__(self, n_input, n_hidden, n_output, activation='sigmoid'):
        super().__init__()

        self.n_inputs = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output

        if(activation == 'sigmoid'):
            self.activation = Sigmoid()
        elif(activation == 'tanh'):
            self.activation == Tanh()
        else:
            raise Exception("Non-linearity not found")

        self.w_ih = Linear(n_input, n_hidden)
        self.w_hh = Linear(n_hidden, n_hidden)
        self.w_ho = Linear(n_hidden, n_output)
        self.parameters += self.w_ih.get_parameters()
        self.parameters += self.w_hh.get_parameters()
        self.parameters += self.w_ho.get_parameters()

    def forward(self, input, hidden):
        from_prev_hidden = self.w_hh.forward(hidden)
        combined = self.w_ih.forward(input) + from_prev_hidden
        new_hidden = self.activation.forward(combined)
        output = self.w_ho.forward(new_hidden)
        return output, new_hidden

    def init_hidden(self, batch_size=1):
        return Tensor(np.zeros((batch_size,self.n_hidden)), autograd=True)

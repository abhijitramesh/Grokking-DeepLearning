import numpy as np


class Tensor(object):

    def __init__(self,data,
                autograd=False,
                creators=None,
                creation_op=None,
                id=None):
        self.data = np.array(data)
        self.creation_op = creation_op
        self.creators = creators
        self.grad = None
        self.autograd = autograd
        self.childern = {}

        if(id is None):
            id = np.random.randint(0,100000)
        self.id = id

        if(creators is not None):
            for c in creators:
                if(self.id not in c.childern):
                    c.childern[self.id] = 1
                else:
                    c.childern[self.id] += 1
    def all_childern_grads_accounted_for(self):
        for id,cnt in self.childern.items():
            if(cnt !=0 ):
                return False
        return True

    def backward(self,grad=None,grad_origin=None):
        if(self.autograd):
            if(grad_origin is not None):
                if(self.childern[grad_origin.id]==0):
                    raise Exception("cannto backprop more than once")
                else:
                    self.children[grad_origin.id] -= 1

        if(self.grad is None):
            self.grad = grad
        else:
            self.grad += grad
        if(self.creators is not None and 
            (self.all_childern_grads_accounted_for()or grad_origin is None)):
                if(self.creation_op =="add"):
                    self.creators[0].backward(grad)
                    self.creators[1].backward(grad)

    def __add__(self,other):
        if(self.autograd and other.autograd):
            return Tensor(self.data+other.data,
                          autograd=True,
                      creators=[self,other],
                      creation_op="add")
        return Tensor(self.data+other.data)


    def __repr__(self):
        return str(self.data.__repr__())

    def __str__(self):
        return str(self.data.__str__())




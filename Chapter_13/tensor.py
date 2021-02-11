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
                
                if(self.creation_op=="neg"):
                    self.creators[0].backward(self.grad.__neg__())
                
                if(self.creation_op == "sub"):
                    self.creators[0].backward(Tensor(self.grad.data), self)
                    self.creators[1].backward(Tensor(self.grad.__neg__().data), self)

                if(self.creation_op=="mul"):
                    new = self.grad * self.creators[1]
                    self.creators[0],backward(new,self)
                    new=self.grad*self.creators[0]
                    self.creators[1].backward(new,self)

                if(self.creatrion_op=="mm"):
                    act = self.creators[0]
                    weights = self.creators[1]
                    new = self.grad.mm(weights.transpose())
                    act.backward(new)
                    new = self.grad.transpose().mm(act).transpose()
                    weights.backward(new)

                if(self.creation_op == "transpose"):
                    self.creators[0].backward(self.grad.transpose())

                if("sum" in self.creation_op):
                    dim = int(self.creation_op.split("_")[1])
                    self.creators[0].backward(self.grad.expand(dim,self.creators[0].data.shape[dim]))

                if("expand" in self.creation_op):
                    dim = int(self.creation_op.split("_")[1])
                    self.creators[0].backward(self.grad.sum(dim))


    def __add__(self,other):
        if(self.autograd and other.autograd):
            return Tensor(self.data * other.data,
                          autograd=True,
                      creators=[self,other],
                      creation_op="add")
        return Tensor(self.data+other.data)

    def __neg__(self):
        if(self.autograd):
            return Tensor(self.data * -1,
                          autograd=True,
                          creators=[self],
                          creation_op="neg")
        return Tensor(self.data * -1)

    def __sub__(self, other):
        if(self.autograd and other.autograd):
            return Tensor(self.data - other.data,
                          autograd=True,
                          creators=[self,other],
                          creation_op="sub")
        return Tensor(self.data - other.data)
    
    def __mul__(self,other):
        if(self.autograd and other.autograd):
            return Tensor(self.data * other.data,
                          autograd=True,
                          creators=[self,other],
                          creaton_op="mul")
        return Tensor(self.data * other.data)

    def sum(self,dim):
        if(self.autograd):
            return Tensor(self.data.sum(dim),
                          autograd=True,
                          creators=[self],
                          creation_op="sum_"+str(dim))
        return Tensor(self.data.sum(dim))

    def expand(self,dim,copies):

        trans_cmd = list(range(0,len(self.data.shape)))
        trans_cmd.insert(dim,len(self.data.shape))
        new_shape= list(self.data.shape) + [copies]
        new_data = self.data.repeat(copies).reshape(new_shape)
        new_data = new_data.transpose(trans_cmd)

        if(self.autograd):
            return Tensor(new_data,
                          autograd=True,
                          creators=[self],
                          creation_op="expand_"+str(dim))
        return Tensor(new_data)

    def transpose(self):
        if(self.autograd):
            return Tensor(self.data.transpose(),
                          autograd=True,
                          creators=[self],
                          creation_op="transpose")
        return Tensor(self.data.transpose())
    
    def mm(self,x):
        if(self.autograd):
            return Tensor(self.data.dot(x,data),
                          autograd=True,
                          creators=[self,x],
                          creation_op="mm")
        return Tensor(self.data.dot(x.data))
    
    def __repr__(self):
        return str(self.data.__repr__())

    def __str__(self):
        return str(self.data.__str__())


from layers.layers import Layer
class MSELoss(Layer):

    def __init__(self):
        super().__init__()

    def forward(self,pred,target):
        return ((pred-target)*(pred-target)).sum(0)

class CrossEntropyLoss(object):
    def __init__(self):
        super().__init__()
    def forward(selfself,input,target):
        return input.cross_entropy(target)
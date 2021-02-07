import numpy as np

a = np.array([1,2,3])
b = np.array([0.1,0.2,0.3])
c = np.array([-1,-0.5,0])
d = np.array([0,0,0])

identity = np.eye(3)
print(identity)

this  = np.array([2,4,6])
movie = np.array([10,10,10])
rocks = np.array([1,1,1])

print(this+movie+rocks)
print((this.dot(identity)+movie).dot(identity)+rocks)




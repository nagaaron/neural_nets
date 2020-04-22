import numpy as np

a  = np.array([
    [1,2,3],
    [4,5,6]
])

b  = np.array([
    [10,20],
    [30,40],
    [50,60]
])

print(a.shape)
print(np.sum(a))
print(np.sum(a,axis = 0))
print(np.sum(a,axis = 1))

print(np.mean(a))
print(np.mean(a,axis = 0))
print(np.mean(a,axis = 1))

print(a.reshape(6,1))
print(a.reshape(6,-1)) #trick sums
print(a.T)

print(a@b)

a  = np.array([
    [1,2,3,4],
    [1,2,3,4]
])
b  = np.array([1,1,0,0], dtype = np.uint8) # reduces bit storage
print(b.dtype)
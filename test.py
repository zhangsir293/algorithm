import numpy as np 
x = np.array([[  0,  1,  2],[  3,  4,  5],[  6,  7,  8],[  9,  10,  11]])
z = x[::2,1::]
r1=np.array([[0,1]])
r2=np.array([[3,3]])
rows = np.array([[0,1],[3,3]])
cols = np.array([[0,2],[0,2]]) 
y = x[rows,cols]
np.reshape(x,(4,3))
print(x)
print(x[1,2])



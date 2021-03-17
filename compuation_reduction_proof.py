import numpy as np
x = np.array([[15,2,3],[35,7,23],[121, 788, 1]])

C = np.matmul(x,x.T)
_C = np.matmul(x.T,x)

w,v = np.linalg.eig(C)
_w,_v = np.linalg.eig(_C)
W,V = np.linalg.eigh(_C)

# W=np.flip(W)
# V=np.flip(V)

# some steps to get back eigenvec from _v to v
_v = np.matmul(x, _v)
V = np.matmul(x, V)

eig_pairs = [(w[index], v[:,index]) for index in range(len(w))]
_eig_pairs = [(_w[index], _v[:,index]/np.linalg.norm(_v[:,index])) for index in range(len(_w))]
Eig_Pairs = [(W[index], V[:,index]/np.linalg.norm(V[:,index])) for index in range(len(W))]

eig_pairs.sort(reverse=True)
_eig_pairs.sort(reverse=True)
Eig_Pairs.sort(reverse=True)

print(eig_pairs)
print(_eig_pairs)
print(Eig_Pairs)

print(eig_pairs[1][1])
# a= np.array([-0.00535458, -0.01542065, -0.99986676])
# b= np.array([  -4.2694304 ,  -12.29554153, -797.23616892])

# anorm = a / np.linalg.norm(a)
# bnorm = b / np.linalg.norm(b)

# print(anorm)
# print(bnorm)
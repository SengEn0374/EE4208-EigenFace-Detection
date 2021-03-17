import numpy as np

x = np.array([[1,4,8],[2,1,8],[3,2,1]])
# print(x)

avg = x.mean(axis=1).reshape(-1,1)
# print(avg)

adjusted = x - avg
# print(adjusted)

c = (1/3)*np.matmul(adjusted, adjusted.T)
# print(c)
w,v = np.linalg.eigh(c)
print(w)
# print('v:', v)


x_T = x.T
# print(x_T)

avg_T = x_T.mean(axis=0)
#print(avg_T)

adjus_t = x_T - avg_T
#print(adjus_t)

c_T = (1/3)*np.matmul(x_T, x_T.T)
w_T,v_T = np.linalg.eigh(c_T)
# print('v_T:', v_T)
print(w_T)

v_eq = np.matmul(x, v_T)
# print('v_eq:', v_eq)

v_test = (v_eq / 1.46326212) * 0.3045548
# print(v_test)


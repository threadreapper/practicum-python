import numpy as np

my_array = np.arange(10,70,2)
print(my_array)

A = np.reshape(my_array,(6,5)).transpose()
A = A * 2 - 5
print(A)

B = np.random.randint(0, 10, 18).reshape(6,3)
print(B)

a = A.sum(axis=1)
b = B.sum(axis=0)
print(np.size(a),np.size(b))

multiplication = A @ B

A = np.delete(A, 3, axis=1)
new_cols = np.random.randint(10, 20, 18).reshape(6, 3)
B = np.hstack((B, new_cols))

det_A = np.linalg.det(A)
det_B = np.linalg.det(B)
if ((det_A != 0) and (det_B != 0)):
    inv_A = np.linalg.inv(A)
    inv_B = np.linalg.inv(B)
else:
    print("невозможно найти обратные матрицы")   
    
A = np.linalg.matrix_power(A, 6)
B = np.linalg.matrix_power(A, 14)

# мой вариант - 2
coefficients = np.array((2, -5, 1, 0, 1, -1, 13, 0, 3, -2, -2, -4, 4, 0, 2.7, -1.3)).reshape(4, 4)
vector = np.array((-4, 2.6, 1, -2)).reshape(4, 1)
print(np.linalg.solve(coefficients, vector))

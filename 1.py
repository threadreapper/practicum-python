import numpy as np

def printMat(matrix):
    print("\n".join(" ".join(f"{x:5.1f}" for x in row) for row in matrix), "\n")

def printVect(vect):
    print(" ".join(f"{x[0]:.1f}" for x in vect))
    
my_array = np.arange(10,70,2)
print(" ".join(map(str, my_array)), "\n")

A = np.reshape(my_array,(6,5)).transpose()
A = A * 2 - 5
printMat(A)

B = np.random.randint(0, 10, 18).reshape(6,3)
printMat(B)

a = A.sum(axis=1)
b = B.sum(axis=0)
print(np.size(a),np.size(b))

multiplication = A @ B
printMat(multiplication)

A = np.delete(A, 3, axis=1)
new_cols = np.random.randint(10, 20, 18).reshape(6, 3)
B = np.hstack((B, new_cols))
printMat(A)
printMat(B)

det_A = np.linalg.det(A)
det_B = np.linalg.det(B)
print(det_A, det_B)

if det_A != 0:
    inv_A = np.linalg.inv(A)
    printMat(inv_A)
else:
    print("невозможно найти обратную матрицу для A")   

if det_B != 0:
    inv_B = np.linalg.inv(B)
    printMat(inv_B)
else:
    print("невозможно найти обратную матрицу для B")

A = np.linalg.matrix_power(A, 6)
B = np.linalg.matrix_power(A, 14)
printMat(A)
printMat(B)

# мой вариант - 2
coefficients = np.array((2, -5, 1, 0, 1, -1, 13, 0, 3, -2, -2, -4, 4, 0, 2.7, -1.3)).reshape(4, 4)
vector = np.array((-4, 2.6, 1, -2)).reshape(4, 1)
printVect(np.linalg.solve(coefficients, vector))

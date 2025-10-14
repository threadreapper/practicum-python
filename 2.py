# мой вариант - 2
import numpy as np
import scipy as sp
import sympy as sym
from scipy.optimize._numdiff import approx_derivative

def f(x): return np.log(x**0.5)
x0 = np.array([2])
a = 1
b = 6

def f2(x): return ((x[0] - 3)**2 + (x[1] - 1)**2)
cons = (
    {'type': 'ineq', 'fun': lambda x: -2*x[0] + x[1] - 5},
    {'type': 'ineq', 'fun': lambda x: 3*x[1] - 10},
)
bnds = ((0, None), (0, None))
x0 = np.array([3, 4])

df_dx = approx_derivative(f, x0, rel_step=1e-6)
print("f'(2) = ", df_dx[0])

def f_prime(x): return approx_derivative(f, x, rel_step=1e-6)[0]
d2f_dx2 = approx_derivative(f_prime, x0, rel_step=1e-6)
print("f''(2) = " , d2f_dx2[0])

x = sym.Symbol('x')
f_sym = sym.log(x**0.5)
diff = sym.diff(f_sym, x)
print("f'(x) = ", diff)

# в scipy нет функции для вычисления интеграла методом прямоугольников 
# (там квадратурная интеграция), поэтому я написал свою
def rectangle_integral(f, a, b, n=1000):
    dx = (b - a) / n
    x = np.linspace(a, b - dx, n)
    return np.sum(f(x) * dx)
integrated_rectangle = rectangle_integral(f, a, b)
print(integrated_rectangle)

# а это уже через scipy
integrated_scipy = sp.integrate.quad(f, a, b)
print(integrated_scipy)

indefinite_integral = sym.integrate(f_sym, x)
print("∫f dx =", indefinite_integral)

opt = sp.optimize.minimize(f2, x0, constraints=cons, bounds=bnds)
if opt.success:
    print("оптимальное решение: ", opt.x)
    print("оптимальное значение: ", opt.fun)

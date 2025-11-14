import numpy as np
import scipy as sp
import sympy as sym
from scipy.optimize._numdiff import approx_derivative

def f(x): return np.log(x**0.5)
x0 = np.array([2])
a = 1
b = 6

df_dx = approx_derivative(f, np.array([2]), rel_step=1e-6)
print(f"f'(2) =  {df_dx[0]:.2f}")
def f_prime(x): return approx_derivative(f, x, rel_step=1e-6)[0]
d2f_dx2 = approx_derivative(f_prime, np.array([2]), rel_step=1e-6)
print(f"f''(2) =  {d2f_dx2[0]:.2f}")

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
print(f"Интеграл (прямоугольники): {integrated_rectangle:.2f}")

# это уже в scipy
integrated_scipy = sp.integrate.quad(f, a, b)
print(f"Интеграл (scipy.quad): {integrated_scipy[0]:.2f}, {integrated_scipy[1]:.2e}")

indefinite_integral = sym.integrate(f_sym, x)
print("∫f dx =", indefinite_integral)

def f2(x): return ((x[0] - 3)**2 + (x[1] - 1)**2)
cons = (
    {'type': 'ineq', 'fun': lambda x: -2*x[0] + x[1] - 5},
    {'type': 'ineq', 'fun': lambda x: 3*x[1] - 10},
)
bnds = ((0, None), (0, None))
x0_opt = np.array([3, 4])

opt = sp.optimize.minimize(f2, x0_opt, constraints=cons, bounds=bnds)
if opt.success:
    solution_str = ", ".join(f"{val:.2f}" for val in opt.x)
    print(f"Оптимальное решение: {solution_str}")
    print(f"Оптимальное значение: {opt.fun:.2f}")

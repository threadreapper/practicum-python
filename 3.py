import numpy as np
import scipy.linalg
from scipy import stats


np.set_printoptions(precision=2, suppress=True, floatmode='fixed', linewidth=100)
def print_formatted(title, arr):
    print(title)
    matrix_str = np.array2string(arr, separator=' ', formatter={'float_kind': lambda x: "%.2f" % x})
    print(matrix_str.strip()[1:-1].strip())
    print("-" * 60)

A = np.array((2, -5, 1, 0, 1, -1, 13, 0, 3, -2, -2, -4, 4, 0, 2.7, -1.3)).reshape(4, 4)
P, L, U = scipy.linalg.lu(A)
det_L = np.prod(np.diag(L))
det_U = np.prod(np.diag(U))
det_P = np.linalg.det(P)
det_total = det_P * det_L * det_U

print_formatted("Матрица А:", A)
print_formatted("Матрица L (Lower):", L)
print_formatted("Матрица U (Upper):", U)
print_formatted("Матрица P (Permutation):", P)

N = 100
low_bound = 0
high_bound = 20

equal = np.random.randint(low_bound, high_bound, N)
mean_loc = (high_bound + low_bound) / 2
scale = 3
normal = np.random.normal(loc=mean_loc, scale=scale, size=N).astype(int)
normal = np.clip(normal, low_bound, high_bound - 1)
samples = {"Равномерная": equal, "Нормальная": normal}

results = {}
for name, data in samples.items():
    mode_val = stats.mode(data, keepdims=True)[0][0]
    results[name] = {
        "Среднее": np.mean(data),
        "Мода": mode_val,
        "Медиана": np.median(data),
        "Минимум": np.min(data),
        "Максимум": np.max(data),
        "Стд. отклонение": np.std(data)
    }
for name, res in results.items():
    print(f"{name} Выборка (100 целых чисел в [0, 19])")
    for stat, value in res.items():
        print(f"  * {stat}: {value:.2f}")
    print("-" * 60)

observed_counts_unif = np.bincount(equal, minlength=high_bound)[low_bound:high_bound]
num_bins = high_bound - low_bound
expected_counts = np.full_like(observed_counts_unif, N / num_bins, dtype=float)
chi2_stat, p_value = stats.chisquare(f_obs=observed_counts_unif, f_exp=expected_counts)

if p_value < 0.05:
    print("P < 0.05. Нулевая гипотеза отвергается. Распределение не является равномерным.")
else:
    print("P > 0.05. У нас нет оснований отвергать нулевую гипотезу. Распределение согласуется с равномерным")

print("-" * 60)

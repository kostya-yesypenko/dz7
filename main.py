import numpy as np
import sympy as sp
from scipy.optimize import minimize_scalar


def fast_gradient(func, x):
    x1, x2 = sp.symbols('x1 x2')
    gradient_func = sp.lambdify((x1, x2), [func(x1, x2).diff(var) for var in (x1, x2)])
    return np.array(gradient_func(x[0], x[1]))


def calculate_alpha(func, x):
    return minimize_scalar(
        lambda alpha: func(x[0] - alpha * fast_gradient(func, x)[0], x[1] - alpha * fast_gradient(func, x)[1]),
        method='golden').x


def steepest_descent(func, x, eps1, eps2, M):
    k = 0

    while True:
        grad = fast_gradient(func, x)

        if np.linalg.norm(grad) < eps1 or k >= M:
            return x

        alpha = calculate_alpha(func, x)
        x_next = x - alpha * grad

        print(
            f'Ітерація{k} xk={np.round(x, 2)},'
            f'|∇f(xk)|={list(np.round(grad, 2))}, '
            f'ak={np.round(alpha, 2)}, '
            f'x(k+1)={list(np.round(x_next, 2))}, '
            f'|x(k+1) − xk|={np.round(np.linalg.norm(x_next - x), 2)}, '
            f'|f(x(k+1) - f(xk))|={np.round(abs(func(*x_next) - func(*x)), 4)}')

        if np.linalg.norm(x_next - x) < eps2 and abs(func(*x_next) - func(*x)) < eps2:
            return x_next

        x = x_next
        k += 1


def func(x1, x2):
    return (x1 ** 2 + x2 ** 2 - 1) ** 2 + (x1 + x2 - 1) ** 2


x0 = np.array([3, 0])
epsilon1 = 0.1
epsilon2 = 0.15
max_iterations = 10

print('Функція f(x1, x2) = (x1^2 + x2^2 - 1)^2 + (x1 + x2 - 1)^2\n\
x^0 = (3;0)^T, ak = 0.5, epsilon1 = 0.1, epsilon2 = 0.15, M = 10\n')
print('Метод найшвидшого градієнтного спуску')
result = steepest_descent(func, x0, epsilon1, epsilon2, max_iterations)

print(f'\nОптимальне значення x1 = {round(result[0], 4)} x2 = {round(result[1], 4)}')
print(f'Мінімальне значення функції f(x1, x2) = {round(func(*result), 4)}')






#ИИ ЛР1---------------------------> Отредактировать

#Функция "подставка для яиц"
#Функция Стыбинского-Танга
#Запрограммировать собственную реализацию классического градиентного спуска
#Запрограммировать пайлайн тестирования алгоритма оптимизации
#Визуализации функции и точки оптимума
#Вычисление погрешности найденного решения в сравнение с аналитическим для нескольких запусков
#Визуализации точки найденного решения (можно добавить анимацию на плюс балл)
#Запрограммировать метод вычисления градиента
#Передача функции градиента от пользователя
#Символьное вычисление градиента (например с помощью sympy) (на доп балл)
#Численная аппроксимация градиента (на доп балл)
#Запрограммировать одну моментную модификацию и протестировать ее
#Запрограммировать одну адаптивную модификацию и протестировать ее
#Запрограммировать метод эволюции темпа обучения и/или метод выбора начального приближения и протестировать их

import numpy as np
import math
import matplotlib.pyplot as plt
import sympy as sp
from mpl_toolkits.mplot3d import Axes3D  
from matplotlib.animation import FuncAnimation


def Egg_holder(x, y):
    return - (y + 47) * np.sin(np.sqrt(np.abs(x / 2 + (y + 47)))) - x * np.sin(np.sqrt(np.abs(x - (y + 47))))


def Stib_tang(x, n):
    sum = 0
    for i in range(n):
        sum += x[i]**4 - 16 * x[i]**2 + 5 * x[i]
    return sum / 2


def gradient_egg(x):
    x_val, y_val = x  
    term1 = -(y_val + 47) * np.cos(np.sqrt(np.abs(x_val / 2 + (y_val + 47)))) / (2 * np.sqrt(np.abs(x_val / 2 + (y_val + 47))))
    term2 = -np.sin(np.sqrt(np.abs(x_val - (y_val + 47)))) - x_val * np.cos(np.sqrt(np.abs(x_val - (y_val + 47)))) / (2 * np.sqrt(np.abs(x_val - (y_val + 47))))
    df_dx = term1 + term2
    
    df_dy = -np.sin(np.sqrt(np.abs(x_val / 2 + (y_val + 47)))) - (y_val + 47) * np.cos(np.sqrt(np.abs(x_val / 2 + (y_val + 47)))) / (2 * np.sqrt(np.abs(x_val / 2 + (y_val + 47))))
    
    return np.array([df_dx, df_dy])


def gradient_stib(x, n):
    grad = np.zeros(n)
    for i in range(n):
        grad[i] = 4 * x[i]**3 - 32 * x[i] + 5
    return grad


def adam_optimizer(grad_func, x_init, eta, beta1, beta2, epsilon, n_iter, decay_lambda, *args):
    x = np.array(x_init, dtype=float)
    m = np.zeros_like(x)  # Переменная момента
    v = np.zeros_like(x)  # Накопление квадратов градиента
    # Список для хранения точек пути оптимизации
    path = [x.copy()]
    
    for t in range(1, n_iter + 1):
        eta_t = eta / (1 + decay_lambda * t)  # Эволюция темпа обучения
        grad = np.array(grad_func(x, *args))  # Градиент
        
        m = beta1 * m + (1 - beta1) * grad  # Момент
        v = beta2 * v + (1 - beta2) * grad**2  # Адаптивность
        
        m_hat = m / (1 - beta1**t)  # Поправка на смещение
        v_hat = v / (1 - beta2**t)  # Поправка на смещение
        
        x = x - (eta_t / (np.sqrt(v_hat) + epsilon)) * m_hat  # Шаг обновления
        # Сохраняем точку для визуализации
        path.append(x.copy())
    return x, path


x_init_egg = [0, 0]  
eta = 0.01
beta1, beta2 = 0.9, 0.999
epsilon = 1e-8
decay_lambda = 0.001
n_iter = 1000

result_egg, path_egg = adam_optimizer(gradient_egg, x_init_egg, eta, beta1, beta2, epsilon, n_iter, decay_lambda)


x_vals = np.linspace(-512, 512, 100)
y_vals = np.linspace(-512, 512, 100)
X, Y = np.meshgrid(x_vals, y_vals)
Z = Egg_holder(X, Y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)


path_egg = np.array(path_egg)
ax.plot(path_egg[:, 0], path_egg[:, 1], Egg_holder(path_egg[:, 0], path_egg[:, 1]), color='r', marker='o')


min_point = np.array([0, 0])  
ax.scatter(min_point[0], min_point[1], Egg_holder(min_point[0], min_point[1]), color='r', s=100, label='Minimum')

ax.set_title("Оптимизация функции Egg Holder")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

plt.legend()
plt.show()

print(f"Результат оптимизации Egg Holder: x = {result_egg[0]}, y = {result_egg[1]}")


n = 3  
x_init_stib = np.zeros(n)  
result_stib, path_stib = adam_optimizer(gradient_stib, x_init_stib, eta, beta1, beta2, epsilon, n_iter, decay_lambda, n)


n = 3  
x_vals = np.linspace(-5, 5, 100)  
y_vals = np.linspace(-5, 5, 100)  
X, Y = np.meshgrid(x_vals, y_vals)


Z = np.array([[Stib_tang([x, y, 0], n) for x in x_vals] for y in y_vals])


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)


ax.set_title("3D график функции Stib Tang")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("z")


ax.scatter(result_stib[0], result_stib[1], result_stib[2], color='r', s=100, label='Result (Optimal Point)', marker='o')

plt.show()

print(f"Результат оптимизации Stib Tang: x = {result_stib}")


x, y = sp.symbols('x y')
f = -(y + 47) * sp.sin(sp.sqrt(abs(x / 2 + (y + 47)))) - x * sp.sin(sp.sqrt(abs(x - (y + 47))))
grad_egg = [sp.diff(f, var) for var in (x, y)]


x = sp.symbols('x0:%d' % n)  
f = sum(x[i]**4 - 16 * x[i]**2 + 5 * x[i] for i in range(n)) / 2
grad_stib = [sp.diff(f, xi) for xi in x]


itr = 3
grad_egg_num = []
grad_egg_sym = []
grad_stib_num = []
grad_stib_sym = []

for i in range(itr):
    result_egg, path_egg = adam_optimizer(gradient_egg, x_init_egg, eta, beta1, beta2, epsilon, n_iter, decay_lambda)
    result_stib, path_stib = adam_optimizer(gradient_stib, x_init_stib, eta, beta1, beta2, epsilon, n_iter, decay_lambda, n)
    res_egg, path_egg = adam_optimizer(grad_egg, x_init_egg, eta, beta1, beta2, epsilon, n_iter, decay_lambda)
    res_stib, path_stib = adam_optimizer(grad_stib, x_init_stib, eta, beta1, beta2, epsilon, n_iter, decay_lambda, n)
    grad_egg_num.append(result_egg)
    grad_egg_sym.append(result_stib)
    grad_stib_num.append(res_egg)
    grad_stib_sym.append(res_stib)

print(grad_egg_num)
print(grad_egg_sym)
print(grad_stib_num)
print(grad_stib_sym)

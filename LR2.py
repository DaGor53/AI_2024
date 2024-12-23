#ИИ LR2 --------------> Перепроверить и затестить

#В Pygmo запрогроммировать две своих тестовых функции и найти их оптимум
#3 разными алгоритмами доступными в библиотеке и получить таблицу сравнения

#Функции Функция "подставка для яиц" (Eggholder function) 
#Функция Стыбинского-Танга

import pygmo as pg
import math


# Функция Egg Holder
class egg_holder:
    def fitness(self, x):  #x - массив с двумя ячейками под x и y
        # Функция Egg Holder для двумерного пространства
        return [(-1 * (x[1] + 47) * math.sin((abs(x[0]/2 + (x[1] + 47))**0.5))) - 
                x[0] * math.sin((abs(x[0] - (x[1] + 47))**0.5))]

    def get_bounds(self):
        # Ограничения на параметры (например, [-512, 512] для обоих x и y)
        return ([-512, -512], [512, 512])

# Функция Stib Tang
class stib_tang:
    def __init__(self, n=3):
        self.n = n  # Размерность вектора x

    def fitness(self, x):
        sum = 0
        for i in range(self.n):
            k = x[i]**4 - 16 * x[i]**2 + 5 * x[i]
            sum += k
        return [sum / 2]

    def get_bounds(self):
        # Ограничения на параметры (например, [-5, 5] для каждого x_i)
        return ([-5] * self.n, [5] * self.n)

# Создание задач
egg_holder_problem = pg.problem(egg_holder())
stib_tang_problem = pg.problem(stib_tang(n=3))

# Алгоритмы
algos = [
    pg.algorithm(pg.de(gen=100)),  # Дифференциальная эволюция
    pg.algorithm(pg.sade(gen=100)),  #Самоадаптивная дифференциальная эволюция
    pg.algorithm(pg.sga(gen=100))  # Стохастический градиентный спуск
]

# Результаты
results = []

# Применение алгоритмов к обеим функциям
for problem, func_name in [(egg_holder_problem, "Egg Holder"), (stib_tang_problem, "Stib Tang")]:
    for algo in algos:
        pop = pg.population(problem, 10)  # Популяция из 10 индивидуумов
        pop = algo.evolve(pop)  # Эволюция популяции
        results.append({
            "Function": func_name,
            "Algorithm": str(algo).split("(")[0],
            "Best Solution": pop.champion_x,
            "Best Fitness": pop.champion_f[0]
        })

i = 0
# Таблица с результатами
for result in results:
    i+=1
    if i == 1 or i ==4:
        alg_name = "DE: Differential Evolution"
    elif i == 2 or i == 5:
        alg_name = "saDE: Self-adaptive Differential Evolution"
    else:
        alg_name = "SGA: Genetic Algorithm"
    print(f"{i} Function: {result['Function']}\n Algorithm: {alg_name}\n Best Fitness: {result['Best Fitness']}\n Best Solution: {result['Best Solution']}\n")
    
input()


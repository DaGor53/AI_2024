#ИИ LR3--------------------> Отредактировать
import logging
import sys

import sklearn.datasets
import sklearn.linear_model
import sklearn.model_selection
import optuna
import optuna.visualization as vis
import matplotlib.pyplot as plt



def objective(trial):
    wine = sklearn.datasets.load_wine()
    classes = list(set(wine.target))
    train_x, valid_x, train_y, valid_y = sklearn.model_selection.train_test_split(
        wine.data, wine.target, test_size=0.25, random_state=0
    )

    alpha = trial.suggest_float("alpha", 1e-5, 1e-1, log=True)
    penalty = trial.suggest_categorical("penalty", ["l2", "l1", "elasticnet", None]) 
    max_iter = trial.suggest_int("max_iter", 100, 1000)  
    tol = trial.suggest_float("tol", 1e-5, 1e-1, log=True)  
    
    eta0 = trial.suggest_float("eta0", 0.001, 1.0) 
    fit_intercept = trial.suggest_categorical("fit_intercept", [True, False]) 

    clf = sklearn.linear_model.Perceptron(
        alpha=alpha, 
        penalty=penalty, 
        max_iter=max_iter, 
        tol=tol, 
        random_state=0, 
        eta0=eta0, 
        fit_intercept=fit_intercept
    )

    for step in range(100):
        clf.partial_fit(train_x, train_y, classes=classes)

        
        intermediate_value = 1.0 - clf.score(valid_x, valid_y)
        trial.report(intermediate_value, step)

        
        if trial.should_prune():
            raise optuna.TrialPruned()

    return 1.0 - clf.score(valid_x, valid_y)


optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))


best_res = float('inf')

print('Исследование 1')
study = optuna.create_study(sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=20)
best_res = min(best_res, study.best_value)

# График истории оптимизации
fig = vis.plot_optimization_history(study)
fig.show()

# График промежуточных значений
fig = vis.plot_intermediate_values(study)
fig.show()

# График параллельных координат
fig = vis.plot_parallel_coordinate(study)
fig.show()

# График контуров
fig = vis.plot_contour(study)
fig.show()

input()

print('Исследование 2')
study = optuna.create_study(sampler=optuna.samplers.RandomSampler(), pruner=optuna.pruners.HyperbandPruner())
study.optimize(objective, n_trials=20)
best_res = min(best_res, study.best_value)

# График истории оптимизации
fig = vis.plot_optimization_history(study)
fig.show()

# График промежуточных значений
fig = vis.plot_intermediate_values(study)
fig.show()

# График параллельных координат
fig = vis.plot_parallel_coordinate(study)
fig.show()

# График контуров
fig = vis.plot_contour(study)
fig.show()

input()

print('Исследование 3')
study = optuna.create_study(sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.HyperbandPruner())
study.optimize(objective, n_trials=20)
best_res = min(best_res, study.best_value)

# График истории оптимизации
fig = vis.plot_optimization_history(study)
fig.show()

# График промежуточных значений
fig = vis.plot_intermediate_values(study)
fig.show()

# График параллельных координат
fig = vis.plot_parallel_coordinate(study)
fig.show()

# График контуров
fig = vis.plot_contour(study)
fig.show()

input()

print('Исследование 4')
study = optuna.create_study(sampler=optuna.samplers.RandomSampler(), pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=20)
best_res = min(best_res, study.best_value)

# График истории оптимизации
fig = vis.plot_optimization_history(study)
fig.show()

# График промежуточных значений
fig = vis.plot_intermediate_values(study)
fig.show()

# График параллельных координат
fig = vis.plot_parallel_coordinate(study)
fig.show()

# График контуров
fig = vis.plot_contour(study)
fig.show()


print('Лучший результат из всех исследований:', best_res)



'''
Код для создания базы данных
CREATE DATABASE optuna_db;
CREATE USER optuna_user WITH PASSWORD '1111';
ALTER ROLE optuna_user SET client_encoding TO 'utf8';
ALTER ROLE optuna_user SET default_transaction_isolation TO 'read committed';
ALTER ROLE optuna_user SET timezone TO 'UTC';
GRANT ALL PRIVILEGES ON DATABASE optuna_db TO optuna_user;
'''
'''
storage = 'postgresql://optuna_user:1111@localhost:5432/optuna_db'

optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))


best_res = float('inf')

print('Исследование 1')
study = optuna.create_study(study_name="Study_2", storage=storage,sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.MedianPruner(),direction="minimize")
study.optimize(objective, n_trials=20)
best_res = min(best_res, study.best_value)

print('Исследование 2')
study = optuna.create_study(study_name="Study_3", storage=storage, sampler=optuna.samplers.RandomSampler(), pruner=optuna.pruners.HyperbandPruner(),direction="minimize")
study.optimize(objective, n_trials=20)
best_res = min(best_res, study.best_value)

print('Исследование 3')
study = optuna.create_study(study_name="Study_4", storage=storage, sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.HyperbandPruner(),direction="minimize")
study.optimize(objective, n_trials=20)
best_res = min(best_res, study.best_value)

print('Исследование 4')
study = optuna.create_study(study_name="Study_5", storage=storage, sampler=optuna.samplers.RandomSampler(), pruner=optuna.pruners.MedianPruner(),direction="minimize")
study.optimize(objective, n_trials=20)
best_res = min(best_res, study.best_value)


print('Лучший результат из всех исследований:', best_res)
'''

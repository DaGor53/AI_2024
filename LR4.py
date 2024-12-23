import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
from scipy.stats import entropy

# Шаг 1. Генерация случайных данных
np.random.seed(42)
data = np.concatenate([np.random.normal(loc=0, scale=1, size=500), 
                       np.random.normal(loc=5, scale=1.5, size=500)])
data = data.reshape(-1, 1)

# Шаг 2. Метод ядерного сглаживания
kde = KernelDensity(kernel='gaussian', bandwidth=0.5)
kde.fit(data)
x_vals = np.linspace(-5, 10, 1000).reshape(-1, 1)
log_density_kde = kde.score_samples(x_vals)
density_kde = np.exp(log_density_kde)

# Шаг 3. ЕМ-алгоритм 
gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(data)
log_density_gmm = gmm.score_samples(x_vals)
density_gmm = np.exp(log_density_gmm)


plt.figure(figsize=(10, 6))
plt.hist(data, bins=30, density=True, alpha=0.5, label="Исходные данные")
plt.plot(x_vals, density_kde, label="Метод ядерного сглаживания", color='blue')
plt.plot(x_vals, density_gmm, label="EM-алгоритм", color='green')
plt.legend()
plt.title("Восстановление плотности")
plt.show()

# Шаг 4. Реализация метода Метрополиса-Гастингса
def metropolis_hastings(pdf, n_samples, proposal_std=1.0, x_init=0.0):
    samples = []
    x = x_init
    for _ in range(n_samples):
        x_new = np.random.normal(x, proposal_std)
        acceptance_ratio = pdf(x_new) / pdf(x)
        if np.random.rand() < acceptance_ratio:
            x = x_new
        samples.append(x)
    return np.array(samples)

# Функция плотности из ядерного сглаживания
pdf_kde = lambda x: np.exp(kde.score_samples(np.array([[x]])))[0]


samples_mh = metropolis_hastings(pdf_kde, 1000)

# Шаг 5. Метод Гиббса
def gibbs_sampling(pdf, n_samples, x_init=0.0):
    samples = [x_init]
    for _ in range(n_samples - 1):
        x_new = np.random.normal(samples[-1], 1.0)
        samples.append(x_new)
    return np.array(samples)

samples_gibbs = gibbs_sampling(pdf_kde, 1000)


plt.figure(figsize=(10, 6))
plt.hist(data, bins=30, density=True, alpha=0.5, label="Исходные данные")
plt.hist(samples_mh, bins=30, density=True, alpha=0.5, label="Метрополис-Гастингс", color='red')
plt.hist(samples_gibbs, bins=30, density=True, alpha=0.5, label="Гиббс", color='purple')
plt.legend()
plt.title("Сравнение наборов точек")
plt.show()

# Шаг 6. Блуждания в 3D
from mpl_toolkits.mplot3d import Axes3D

samples_3d = metropolis_hastings(pdf_kde, 1000, proposal_std=0.5, x_init=0.0)
samples_3d = samples_3d.reshape(-1, 1)
x_3d = samples_3d[:, 0]
y_3d = np.random.normal(0, 1, size=len(samples_3d))
z_3d = np.random.normal(0, 1, size=len(samples_3d))

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_3d, y_3d, z_3d, color='blue', alpha=0.6, label='Метрополис-Гастингс (3D)')
ax.set_title("Блуждания в 3D")
plt.legend()
plt.show()

# Шаг 7. KL-дивергенция
hist_data, _ = np.histogram(data, bins=30, density=True)
hist_mh, _ = np.histogram(samples_mh, bins=30, density=True)
hist_gibbs, _ = np.histogram(samples_gibbs, bins=30, density=True)

kl_mh = entropy(hist_data + 1e-10, hist_mh + 1e-10)
kl_gibbs = entropy(hist_data + 1e-10, hist_gibbs + 1e-10)

print(f"KL-дивергенция для Метрополиса-Гастингса: {kl_mh:.4f}")
print(f"KL-дивергенция для Гиббса: {kl_gibbs:.4f}")

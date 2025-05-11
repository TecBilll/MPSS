import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Вхідні дані
X1 = np.array([0, 0, 0, 1, 1, 2, 2, 2])
X2 = np.array([1.5, 2.5, 3.5, 1.5, 3.5, 1.5, 2.5, 2.5])
Y = np.array([2.3, 4.3, 1.9, 4.8, 4.8, 6.3, 6.3, 7.2])

# Створення матриці ознак
X = np.column_stack((np.ones(len(X1)), X1, X2))

# Знаходження коефіцієнтів МНК
coefficients = np.linalg.lstsq(X, Y, rcond=None)[0]
a0, a1, a2 = coefficients

# Виведення коефіцієнтів
print(f"Методом найменших квадратів знайшла рівняння: y = a_0 + a_1 * x1 + a_2 * x2")
print(f"a_0  = {a0:.6f},")
print(f"a_1  = {a1:.6f},")
print(f"a_2  = {a2:.6f}")

# Функція для прогнозування
def predict(x1, x2):
    return a0 + a1 * x1 + a2 * x2

# Обчислення значення функції в заданій точці
x1_test, x2_test = 1.5, 3
y_pred = predict(x1_test, x2_test)
print(f"Значення функції у точці (x1={x1_test}, x2={x2_test}): {y_pred:.6f}")

# Побудова графіка
x1_range = np.linspace(min(X1), max(X1), 10)
x2_range = np.linspace(min(X2), max(X2), 10)
X1_grid, X2_grid = np.meshgrid(x1_range, x2_range)
Y_grid = predict(X1_grid, X2_grid)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X1, X2, Y, color='red', label='Вхідні дані')
ax.plot_surface(X1_grid, X2_grid, Y_grid, alpha=0.5, color='cyan')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
ax.set_title('Модель лінійної регресії')
plt.show()

# Обчислення R^2
y_pred_all = predict(X1, X2)
R2 = r2_score(Y, y_pred_all)
print(f'Коефіцієнт детермінації R^2: {R2:.6f}')
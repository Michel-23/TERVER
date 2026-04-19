import numpy as np
import matplotlib.pyplot as plt

def linear_regression_lab():

    print("--- Часть 1: Фиксированные x (1, 2, ..., n) ---")
    a = float(input("Введите коэффициент a: "))
    b = float(input("Введите коэффициент b: "))
    sigma = float(input("Введите среднеквадратичное отклонение sigma: "))
    n = int(input("Введите объем выборки n: "))

    X = np.arange(1, n + 1)#Создание массива X
    y_true = a * X + b #Вычисление истинных значений (без шума)
    noise = np.random.normal(0, sigma, n) #Генерация шума
    Y = y_true + noise #Формирование наблюдаемых значений (добавдяем шум)

    # Вычисление коэффициентов (метод Крамера)
    def estimate_coefficients_cramer(x, y):
        n_obs = len(x)   # количество наблюдений (n)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_x2 = np.sum(x ** 2)
        sum_xy = np.sum(x * y)

        det = n_obs * sum_x2 - sum_x * sum_x #Вычисление определителя
        if det == 0:
            raise ValueError("Определитель равен нулю, решение неединственно")
        #Вычисление коэффициентов
        a_star = (n_obs * sum_xy - sum_x * sum_y) / det
        b_star = (sum_x2 * sum_y - sum_x * sum_xy) / det
        return a_star, b_star

    a_star, b_star = estimate_coefficients_cramer(X, Y)
    print(f"Оцененные коэффициенты: a* = {a_star:.4f}, b* = {b_star:.4f}")

    # R^2
    Y_pred = a_star * X + b_star #Предсказанные значения
    ss_res = np.sum((Y - Y_pred) ** 2) #Остаточная сумма квадратов
    ss_tot = np.sum((Y - np.mean(Y)) ** 2) #Общая сумма квадратов
    r_squared = 1 - (ss_res / ss_tot) #Коэффициент детерминации
    print(f"Коэффициент детерминации R^2 = {r_squared:.4f}")

    # Дополнительная выборка
    m = int(input("Введите объем дополнительной выборки m: "))
    X_extra = np.arange(n + 1, n + m + 1)#Генерация новых значений X
    Y_extra_true = a * X_extra + b + np.random.normal(0, sigma, m)#Генерация истинных (наблюдаемых) значений
    Y_extra_pred = a_star * X_extra + b_star#Генерация истинных (наблюдаемых) значений

    # График первой части
    plt.figure(figsize=(10, 6))
    plt.scatter(X, Y, color='blue', label='Обучающая выборка')
    plt.scatter(X_extra, Y_extra_true, color='green', label='Доп. выборка (реальность)')
    plt.plot(np.append(X, X_extra), a_star * np.append(X, X_extra) + b_star,
             color='red', label='Линия регрессии (прогноз)')
    plt.title("Линейная регрессия (фиксированный X)")
    plt.legend()
    plt.grid(True)
    plt.show(block=False) # Окно открылось, код идет дальше
    plt.pause(0.1)        # Даем системе время на отрисовку
    input("Нажмите Enter в консоли, чтобы продолжить...")
    plt.close()           # Закрываем первое окно сами и идем ко второй части

    # """--- Часть 2""": Случайные x на отрезке [t1, t2] ---
    print("\n--- Часть 2: Случайные x на отрезке [t1, t2] ---")
    t1 = float(input("Введите начало отрезка t1: "))
    t2 = float(input("Введите конец отрезка t2: "))

    X_rand = np.random.uniform(t1, t2, n) #Генерация случайных X
    Y_rand = a * X_rand + b + np.random.normal(0, sigma, n) #Генерация наблюдаемых значений

    a_star_rand, b_star_rand = estimate_coefficients_cramer(X_rand, Y_rand) #Оценка коэффициентов
    #Вычисление R^2
    Y_rand_pred = a_star_rand * X_rand + b_star_rand
    r2_rand = 1 - (np.sum((Y_rand - Y_rand_pred) ** 2) / np.sum((Y_rand - np.mean(Y_rand)) ** 2))

    print(f"Оцененные коэффициенты (rand): a* = {a_star_rand:.4f}, b* = {b_star_rand:.4f}")
    print(f"R^2 (rand) = {r2_rand:.4f}")

    plt.figure(figsize=(10, 6))
    plt.scatter(X_rand, Y_rand, color='purple', label='Случайная выборка')
    x_range = np.linspace(t1, t2, 100)
    plt.plot(x_range, a_star_rand * x_range + b_star_rand, color='orange', label='Линия регрессии')
    plt.title("Линейная регрессия (случайный X)")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    linear_regression_lab()
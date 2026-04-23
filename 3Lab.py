import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def estimate_coefficients(x1, x2, y):
        X = np.column_stack((x1, x2, np.ones(len(x1))))
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        return beta[0], beta[1], beta[2]

def run_experiment(a1_val, a2_val, b_val, sigma_val, n_val, t1_val, t2_val, s1_val, s2_val):
    x1 = np.random.uniform(t1_val, t2_val, n_val)
    x2 = np.random.uniform(s1_val, s2_val, n_val)

    y_true = a1_val * x1 + a2_val * x2 + b_val
    y = y_true + np.random.normal(0, sigma_val, n_val)

    a1_est, a2_est, b_est = estimate_coefficients(x1, x2, y)

    y_model = a1_est * x1 + a2_est * x2 + b_est
    ss_res = np.sum((y - y_model) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0

    return a1_est, a2_est, b_est, r2

def print_parameter_table(param_name, values, base_params):
    print(f"\nИзменяем только параметр {param_name}, остальные фиксированы")
    print("-" * 70)
    print(f"{param_name:>10} | {'a1*':>10} | {'a2*':>10} | {'b*':>10} | {'R^2':>10}")
    print("-" * 70)

    for value in values:
        params = base_params.copy()
        params[param_name] = int(value) if param_name == "n" else float(value)

        # Проверка корректности интервалов
        if params["t1"] >= params["t2"] or params["s1"] >= params["s2"]:
            print(f"{str(value):>10} | {'ошибка':>10} | {'ошибка':>10} | {'ошибка':>10} | {'ошибка':>10}")
            continue

        np.random.seed(42)  # чтобы сравнение было воспроизводимым

        a1_est, a2_est, b_est, r2 = run_experiment(
            params["a1"], params["a2"], params["b"], params["sigma"],
            params["n"], params["t1"], params["t2"], params["s1"], params["s2"]
        )

        if param_name == "n":
            print(f"{int(value):>10} | {a1_est:>10.4f} | {a2_est:>10.4f} | {b_est:>10.4f} | {r2:>10.4f}")
        else:
            print(f"{value:>10.4f} | {a1_est:>10.4f} | {a2_est:>10.4f} | {b_est:>10.4f} | {r2:>10.4f}")

    print("-" * 70)


def multiple_linear_regression_lab():
    print("=== Многомерная линейная регрессия ===")

    # Ввод коэффициентов линейного уравнения y(x1, x2) = a1*x1 + a2*x2 + b
    a1 = float(input("Введите коэффициент a1: "))
    a2 = float(input("Введите коэффициент a2: "))
    b = float(input("Введите коэффициент b: "))
    sigma = float(input("Введите среднеквадратичное отклонение sigma: "))
    n = int(input("Введите объем обучающей выборки n: "))

    print("\n--- Интервалы для случайных x1 и x2 ---")
    t1 = float(input("Введите начало отрезка для x1 (t1): "))
    t2 = float(input("Введите конец отрезка для x1 (t2): "))
    s1 = float(input("Введите начало отрезка для x2 (s1): "))
    s2 = float(input("Введите конец отрезка для x2 (s2): "))

    # Генерация обучающей выборки
    x1_train = np.random.uniform(t1, t2, n)
    x2_train = np.random.uniform(s1, s2, n)

    y_true_train = a1 * x1_train + a2 * x2_train + b
    y_train = y_true_train + np.random.normal(0, sigma, n)

    # Оценка коэффициентов
    

    a1_star, a2_star, b_star = estimate_coefficients(x1_train, x2_train, y_train)

    print("\n--- Оцененные коэффициенты ---")
    print(f"a1* = {a1_star:.4f}")
    print(f"a2* = {a2_star:.4f}")
    print(f"b*  = {b_star:.4f}")

    # Идеальные значения на обучающей выборке
    y_pred_train = y_true_train

    # Вычисление R^2 для оцененной модели оставляем как было
    y_model_train = a1_star * x1_train + a2_star * x2_train + b_star
    ss_res = np.sum((y_train - y_model_train) ** 2)
    ss_tot = np.sum((y_train - np.mean(y_train)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else 0

    print(f"\nКоэффициент детерминации R^2 = {r_squared:.4f}")

    # Дополнительная выборка
    m = int(input("\nВведите объем дополнительной выборки m: "))

    x1_test = np.random.uniform(t1, t2, m)
    x2_test = np.random.uniform(s1, s2, m)

    y_true_test = a1 * x1_test + a2 * x2_test + b
    y_test = y_true_test + np.random.normal(0, sigma, m)

    # Идеальные значения на дополнительной выборке
    y_pred_test = y_true_test

      # -----------------------------
    # Для 2D-графика фиксируем x2
    # -----------------------------
    x2_fixed = np.mean(np.concatenate([x2_train, x2_test]))

    x1_line = np.linspace(t1, t2, 200)
    y_line_ideal = a1 * x1_line + a2 * x2_fixed + b   # идеальная прямая

    # -----------------------------
    # График 1: только n элементов
    # -----------------------------
    plt.figure(figsize=(10, 6))
    plt.scatter(x1_train, y_train, color='blue', label='Реальные значения y (n)')
    plt.plot(x1_line, y_line_ideal, color='red',
             label=f'Идеальная прямая при x2 = {x2_fixed:.2f}')
    plt.xlabel("x1")
    plt.ylabel("y")
    plt.title("Обучающая выборка (n элементов)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # -----------------------------
    # График 2: n + m элементов
    # n одним цветом, m другим
    # -----------------------------
    plt.figure(figsize=(10, 6))
    plt.scatter(x1_train, y_train, color='blue', label='Реальные y (первые n)')
    plt.scatter(x1_test, y_test, color='green', label='Реальные y (доп. m)')
    plt.plot(x1_line, y_line_ideal, color='red',
             label=f'Идеальная прямая при x2 = {x2_fixed:.2f}')
    plt.xlabel("x1")
    plt.ylabel("y")
    plt.title("Сравнение реальных значений и идеальной прямой")
    plt.legend()
    plt.grid(True)
    plt.show()

    # -----------------------------
    # График 3: 3D-визуализация
    # n — одним цветом, m — другим
    # -----------------------------
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Обучающая выборка (n)
    ax.scatter(x1_train, x2_train, y_train, color='blue', alpha=0.7, label='Обучающая выборка (n)')

    # Дополнительная выборка (m)
    ax.scatter(x1_test, x2_test, y_test, color='green', alpha=0.7, label='Дополнительная выборка (m)')

    # Сетка для плоскости регрессии
    x1_grid = np.linspace(t1, t2, 30)
    x2_grid = np.linspace(s1, s2, 30)
    X1_grid, X2_grid = np.meshgrid(x1_grid, x2_grid)

    Y_grid = a1_star * X1_grid + a2_star * X2_grid + b_star

    # Плоскость регрессии
    ax.plot_surface(X1_grid, X2_grid, Y_grid, alpha=0.5, cmap='viridis')

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("y")
    ax.set_title("3D-визуализация многомерной линейной регрессии")
    ax.legend()

    plt.show()

   
    # -----------------------------
    # Таблицы изменения параметров
    # -----------------------------
    base_params = {
        "a1": a1,
        "a2": a2,
        "b": b,
        "sigma": sigma,
        "n": n,
        "t1": t1,
        "t2": t2,
        "s1": s1,
        "s2": s2
    }

    delta_t = t2 - t1
    delta_s = s2 - s1

    param_variations = {
        "a1": [int(a1), int(a1 + 100), int(a1 + 500), int(a1 + 1000)],
        "a2": [int(a2), int(a2 + 100), int(a2 + 500), int(a2 + 1000)],
        "b": [int(b), int(b + 100), int(b + 500), int(b + 1000)],
        "sigma": [int(sigma), int(sigma * 50), int(sigma * 100), int(sigma + 300)],
        "n": [int(n), int(n * 10), int(n * 100), int(n * 1000)],
        "t1": [t1, t1 + delta_t / 4, t1 + delta_t / 2, t1 + 3 * delta_t / 4],
        "t2": [t2, t2 + delta_t / 4, t2 + delta_t / 2, t2 + delta_t],
        "s1": [s1, s1 + delta_s / 4, s1 + delta_s / 2, s1 + 3 * delta_s / 4],
        "s2": [s2, s2 + delta_s / 4, s2 + delta_s / 2, s2 + delta_s]
    }

    for param_name, values in param_variations.items():
        print_parameter_table(param_name, values, base_params)
"вывести предельное значени для r^2"
if __name__ == "__main__":
    multiple_linear_regression_lab()
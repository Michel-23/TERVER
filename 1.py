import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# НАСТРОЙКИ ЭКСПЕРИМЕНТА
# ============================================================

CONFIG = {
    # Истинная модель:
    # y = a1*x1 + a2*x2 + b + шум
    "a1_true": 3.0,
    "a2_true": -3.0,
    "b_true": 50.0,

    # Параметры шума и выборки
    "sigma": 15.0,
    "n": 2000,
    # для генерации случайнах точек
    "seed": 42,

    # Отрезки изменения признаков
    "x1_range": (10, 20),
    "x2_range": (10, 20),

    # Шум зависимости x1 от x2
    # Это шум НЕ в y, а шум самой кривой x1=f(x2)
    "dependency_noise": 0.3,

    # Какие зависимости запускать
    "experiments": [
        "linear_asc",
        "linear_desc",
        "quadratic",
        "sinusoidal",
    ],
}


# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ АВТОМАТИЧЕСКИХ ЗАВИСИМОСТЕЙ
# ============================================================

def scale_to_range(z, target_min, target_max):
    z_min = np.min(z)
    z_max = np.max(z)

    if z_max == z_min:
        return np.full_like(z, (target_min + target_max) / 2)

    return target_min + (z - z_min) / (z_max - z_min) * (target_max - target_min)


def get_ranges(config):
    x1_min, x1_max = config["x1_range"]
    x2_min, x2_max = config["x2_range"]
    return x1_min, x1_max, x2_min, x2_max


def true_dependency(x2, dep_type, config):
    x1_min, x1_max, x2_min, x2_max = get_ranges(config)

    x1_mid = (x1_min + x1_max) / 2
    x1_amp = (x1_max - x1_min) / 2

    x2_mid = (x2_min + x2_max) / 2
    x2_width = x2_max - x2_min

    if dep_type == "linear_asc":
        k = (x1_max - x1_min) / x2_width
        m = x1_min - k * x2_min

        x1 = k * x2 + m
        formula = f"x₁ = {k:.3g}·x₂ + {m:.3g}"

    elif dep_type == "linear_desc":
        k = -(x1_max - x1_min) / x2_width
        m = x1_max - k * x2_min

        x1 = k * x2 + m
        formula = f"x₁ = {k:.3g}·x₂ + {m:.3g}"

    elif dep_type == "quadratic":
        A = (x1_max - x1_min) / ((x2_width / 2) ** 2)

        x1 = A * (x2 - x2_mid) ** 2 + x1_min
        formula = f"x₁ = {A:.3g}·(x₂ - {x2_mid:.3g})² + {x1_min:.3g}"

    elif dep_type == "sinusoidal":
        A = x1_amp
        w = 2 * np.pi / x2_width
        shift = x2_min + x2_width / 4

        x1 = x1_mid + A * np.sin(w * (x2 - shift))
        formula = f"x₁ = {A:.3g}·sin({w:.3g}·(x₂ - {shift:.3g})) + {x1_mid:.3g}"

    else:
        raise ValueError(f"Неизвестный тип зависимости: {dep_type}")

    return x1, formula


# ============================================================
# РЕГРЕССИЯ И R²
# ============================================================

def estimate_coefficients(x1, x2, y):
    # оздаем матрицу признаков вида [x1i x2i 1]^т
    X = np.column_stack((x1, x2, np.ones(len(x1))))
    # решаем задачу МНК, где beta = (a1 a2 b) вектор
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return beta[0], beta[1], beta[2]

# теоретическая формула R^2
def theoretical_r2(x1, x2, a1, a2, sigma):
    D_x1 = np.var(x1)
    D_x2 = np.var(x2)
    # тут возвращается матрица из дисперсий и ковариаций и берется её элемент из 1 строки и 2 столбца то есть cov(x1,x2)
    cov_x1_x2 = np.cov(x1, x2, bias=True)[0, 1]

    signal_var = (
        a1**2 * D_x1
        + a2**2 * D_x2
        + 2 * a1 * a2 * cov_x1_x2
    )

    return signal_var / (signal_var + sigma**2)

# тут считается практический и теоретический коэффициент детерминации
def run_experiment(x1, x2, config):
    a1_true = config["a1_true"]
    a2_true = config["a2_true"]
    b_true = config["b_true"]
    sigma = config["sigma"]

    n = len(x1)

    y_true = a1_true * x1 + a2_true * x2 + b_true
    y = y_true + np.random.normal(0, sigma, n)

    a1_est, a2_est, b_est = estimate_coefficients(x1, x2, y)

    y_pred = a1_est * x1 + a2_est * x2 + b_est

    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    r2_practical = 1 - ss_res / ss_tot if ss_tot != 0 else 0
    r2_theoretical = theoretical_r2(x1, x2, a1_true, a2_true, sigma)

    return a1_est, a2_est, b_est, r2_practical, r2_theoretical, y


# ============================================================
# ГЕНЕРАЦИЯ ДАННЫХ
# ============================================================

def generate_dependent_data(dep_type, config):
    n = config["n"]
    x1_min, x1_max, x2_min, x2_max = get_ranges(config)

    x2 = np.random.uniform(x2_min, x2_max, n)

    x1_clean, formula = true_dependency(x2, dep_type, config)

    noise = np.random.normal(0, config["dependency_noise"], n)
    x1 = x1_clean + noise
    x1 = np.clip(x1, x1_min, x1_max)

    return x1, x2, x1_clean, formula


# ============================================================
# АНАЛИЗ ОДНОГО ТИПА ЗАВИСИМОСТИ
# ============================================================

def analyze_dependency(dep_type, config):
    print(f"\n{'=' * 70}")
    print(f"Тип зависимости: {dep_type}")
    print(f"{'=' * 70}")

    x1_min, x1_max, x2_min, x2_max = get_ranges(config)

    x1, x2, _, formula = generate_dependent_data(dep_type, config)

    print(f"Формула: {formula}")
    print(f"Диапазон x₁: [{np.min(x1):.2f}, {np.max(x1):.2f}]")
    print(f"Диапазон x₂: [{np.min(x2):.2f}, {np.max(x2):.2f}]")

    a1_est, a2_est, b_est, r2_practical, r2_theoretical, y = run_experiment(
        x1, x2, config
    )

    corr = np.corrcoef(x1, x2)[0, 1]

    print(
        f"\nИстинные коэффициенты: "
        f"a₁ = {config['a1_true']}, "
        f"a₂ = {config['a2_true']}, "
        f"b = {config['b_true']}"
    )
    print(f"Оценённые коэффициенты: a₁* = {a1_est:.4f}, a₂* = {a2_est:.4f}, b* = {b_est:.4f}")
    print(f"Практический R² = {r2_practical:.6f}")
    print(f"Теоретический R² = {r2_theoretical:.6f}")
    print(f"Корреляция между x₁ и x₂ = {corr:.6f}")

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(x2, x1, alpha=0.6, color="purple")
    plt.xlabel("x₂")
    plt.ylabel("x₁")
    plt.title(f"Зависимость x₁ от x₂\n{formula}")
    plt.xlim(x2_min - 1, x2_max + 1)
    plt.ylim(x1_min - 1, x1_max + 1)
    plt.grid(True)

    x2_sorted = np.linspace(x2_min, x2_max, 500)
    x1_smooth, _ = true_dependency(x2_sorted, dep_type, config)

    plt.plot(
        x2_sorted,
        x1_smooth,
        "r-",
        linewidth=2,
        label="Истинная зависимость без шума"
    )
    plt.legend()

    ax = plt.subplot(1, 2, 2, projection="3d")
    scatter = ax.scatter(x1, x2, y, c=y, cmap="viridis", alpha=0.7)

    x1_grid = np.linspace(x1_min, x1_max, 30)
    x2_grid = np.linspace(x2_min, x2_max, 30)
    X1_grid, X2_grid = np.meshgrid(x1_grid, x2_grid)
    Y_grid = a1_est * X1_grid + a2_est * X2_grid + b_est

    ax.plot_surface(X1_grid, X2_grid, Y_grid, alpha=0.4, color="red")
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.set_zlabel("y")
    ax.set_xlim(x1_min, x1_max)
    ax.set_ylim(x2_min, x2_max)
    ax.set_title(f"3D визуализация\nR² = {r2_practical:.4f}")

    plt.colorbar(scatter, ax=ax, shrink=0.5, label="y")
    plt.tight_layout()
    plt.show()

    return {
        "a₁*": a1_est,
        "a₂*": a2_est,
        "b*": b_est,
        "R² практич.": r2_practical,
        "R² теорет.": r2_theoretical,
        "Корреляция": corr,
    }


# ============================================================
# СРАВНЕНИЕ ВСЕХ ЗАВИСИМОСТЕЙ
# ============================================================

def compare_dependencies(config):
    print("=== Анализ влияния зависимости x₁ от x₂ на качество регрессии ===\n")

    print(f"Область x₁: {config['x1_range']}")
    print(f"Область x₂: {config['x2_range']}")
    print(
        f"Истинная модель: "
        f"y = {config['a1_true']}·x₁ + ({config['a2_true']})·x₂ + {config['b_true']}"
    )
    print(f"Сигма шума: σ = {config['sigma']}")
    print(f"n = {config['n']}")

    results = {}

    names = {
        "linear_asc": "Линейная (возраст.)",
        "linear_desc": "Линейная (убыв.)",
        "quadratic": "Параболическая",
        "sinusoidal": "Синусоидальная",
    }

    for dep_type in config["experiments"]:
        name = names.get(dep_type, dep_type)
        results[name] = analyze_dependency(dep_type, config)

    print("\n" + "=" * 105)
    print("СВОДНАЯ ТАБЛИЦА: ПРАКТИЧЕСКИЙ И ТЕОРЕТИЧЕСКИЙ R²")
    print("=" * 105)
    print(
        f"{'Тип зависимости':<22} | "
        f"{'a₁*':>10} | "
        f"{'a₂*':>10} | "
        f"{'b*':>10} | "
        f"{'R² практ.':>12} | "
        f"{'R² теор.':>12} | "
        f"{'Корр.':>10}"
    )
    print("-" * 105)

    for name, data in results.items():
        print(
            f"{name:<22} | "
            f"{data['a₁*']:>10.4f} | "
            f"{data['a₂*']:>10.4f} | "
            f"{data['b*']:>10.4f} | "
            f"{data['R² практич.']:>12.6f} | "
            f"{data['R² теорет.']:>12.6f} | "
            f"{data['Корреляция']:>10.6f}"
        )

    print("=" * 105)


# ============================================================
# БАЗОВЫЙ СЛУЧАЙ: НЕЗАВИСИМЫЕ x₁ И x₂
# ============================================================

def test_independent_case(config):
    print("\n\n=== Базовый случай: независимые x₁ и x₂ ===")
    print("=" * 70)

    np.random.seed(config["seed"])

    x1_min, x1_max, x2_min, x2_max = get_ranges(config)

    x1 = np.random.uniform(x1_min, x1_max, config["n"])
    x2 = np.random.uniform(x2_min, x2_max, config["n"])

    a1_est, a2_est, b_est, r2_practical, r2_theoretical, _ = run_experiment(
        x1, x2, config
    )

    corr = np.corrcoef(x1, x2)[0, 1]

    print(
        f"Истинная модель: "
        f"y = {config['a1_true']}·x₁ + ({config['a2_true']})·x₂ + {config['b_true']}"
    )
    print(f"Диапазон x₁: {config['x1_range']}")
    print(f"Диапазон x₂: {config['x2_range']}")
    print(f"Корреляция между x₁ и x₂: {corr:.6f}")

    print(f"\nОценённые коэффициенты:")
    print(f"  a₁* = {a1_est:.6f}")
    print(f"  a₂* = {a2_est:.6f}")
    print(f"  b*  = {b_est:.6f}")

    print(f"\nПрактический R² = {r2_practical:.6f}")
    print(f"Теоретический R² = {r2_theoretical:.6f}")
    print("=" * 70)


# ============================================================
# ЗАПУСК
# ============================================================

if __name__ == "__main__":
    np.random.seed(CONFIG["seed"])

    test_independent_case(CONFIG)
    compare_dependencies(CONFIG)
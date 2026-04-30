import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt


# 1. Загружаем датасет
df = pd.read_csv("insurance.csv")

print("Первые 5 строк датасета:")
print(df.head())

print("\nРазмер датасета:")
print(df.shape)


# 2. Целевая переменная
# charges — это медицинские расходы, которые мы хотим предсказать
target = "charges"

# Параметры, влияние которых будем исследовать отдельно
features = ["age", "sex", "bmi", "children", "smoker", "region"]


# 3. Делим индексы на обучающую и тестовую выборку
# Это нужно, чтобы для всех параметров использовались одни и те же строки
train_indices, test_indices = train_test_split(
    df.index,
    test_size=0.2,
    random_state=42
)

results = []


# 4. Проверяем каждый параметр отдельно
for feature in features:
    # Берём только один параметр
    X = df[[feature]]
    y = df[target]

    # Если параметр текстовый, преобразуем его в числовой вид
    X_encoded = pd.get_dummies(X, drop_first=True)

    # Разделяем данные на обучающую и тестовую выборку
    X_train = X_encoded.loc[train_indices]
    X_test = X_encoded.loc[test_indices]

    y_train = y.loc[train_indices]
    y_test = y.loc[test_indices]

    # Создаём и обучаем модель линейной регрессии
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Делаем предсказание
    y_pred = model.predict(X_test)

    # Считаем коэффициент детерминации
    r2 = r2_score(y_test, y_pred)

    # Коэффициент a при x в уравнении y = ax + b
    # Если после преобразования получилось несколько столбцов,
    # например region превратился в несколько регионов,
    # то берём средний модуль коэффициентов.
    if len(model.coef_) == 1:
        a_value = model.coef_[0]
    else:
        a_value = sum(abs(coef) for coef in model.coef_) / len(model.coef_)

    # Свободный член b
    b_value = model.intercept_

    # Сохраняем результат
    results.append({
        "Параметр": feature,
        "a": a_value,
        "b": b_value,
        "R²": r2
    })


# 5. Создаём таблицу значимости параметров
results_df = pd.DataFrame(results)

# Сортируем по R²: чем больше R², тем значимее параметр
results_df = results_df.sort_values(by="R²", ascending=False)

# Добавляем место в рейтинге
results_df.insert(0, "Место", range(1, len(results_df) + 1))

print("\nТаблица значимости параметров:")
print(results_df.to_string(index=False))


# 6. Выводим самый значимый параметр
best_parameter = results_df.iloc[0]

print("\nНаиболее значимый параметр:")
print(f"{best_parameter['Параметр']}")

print(
    f"Для этого параметра коэффициент детерминации R² равен "
    f"{best_parameter['R²']:.3f}."
)

print(
    f"Коэффициент a для этого параметра равен "
    f"{best_parameter['a']:.3f}."
)

print(
    "Чем выше значение R², тем лучше данный параметр отдельно объясняет "
    "изменение медицинских расходов."
)


# 7. Строим график изменения R² по каждому параметру
plt.figure(figsize=(9, 6))

plt.bar(results_df["Параметр"], results_df["R²"])

plt.xlabel("Параметр")
plt.ylabel("Коэффициент детерминации R²")
plt.title("Изменение коэффициента детерминации по отдельным параметрам")
plt.grid(axis="y")

plt.show()
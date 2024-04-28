import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
# Завантаження даних з CSV файлу
data = pd.read_csv('car_prices.csv', sep=',', encoding='utf-8')

# Групування даних за роками та обчислення суми sellingprice
sellingprice_by_year = data.groupby('year')['sellingprice'].sum().reset_index()

# Виведення результату
print(sellingprice_by_year)

data_2016 = data[data['year'] == 2016]

# Підготовка даних для моделі
X = sellingprice_by_year[['year']]
y = sellingprice_by_year['sellingprice']


# Побудова моделі випадкового лісу
rf_model = RandomForestRegressor()
rf_model.fit(X, y)

# Прогнозування суми продажів для 2016 року
predicted_sales_2016 = rf_model.predict([[2016]])
print("Прогнозована сума прибутків за продажі моделей 2016 року:", predicted_sales_2016)

y_pred_train = rf_model.predict(X)
r2 = r2_score(y, rf_model.predict(X))
print("R2 Score:", r2)
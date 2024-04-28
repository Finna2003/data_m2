import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# Завантаження даних
data = pd.read_csv('car_prices.csv', sep=',', encoding='utf-8', nrows=10000)

# Видалення непотрібних стовпців
data = data[['year', 'body', 'transmission', 'condition', 'odometer', 'sellingprice']]

# Додавання нового стовпця "десетиліття"
data['decade'] = data['year'].astype(str).str[2]
data['decade'] = data['decade'].astype(int)
print(data.head())
# Розділення даних на вхідні ознаки (X) та цільову змінну (y)
X = data[['sellingprice', 'condition', 'odometer']]
y = data['decade']

# Розділення на навчальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Побудова моделі випадкового лісу
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)

# Прогнозування та оцінка моделі
y_pred = rf_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print("Accuracy:", accuracy)
print("F1 Score:", f1)

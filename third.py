import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# Завантаження даних з CSV файлу
data = pd.read_csv('car_prices.csv', sep=',', encoding='utf-8')

# Вибірка потрібних стовпців (рік та пробіг автомобіля)
X = data[['year', 'sellingprice']]

# Обробка пропущених значень
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Побудова моделі K-Means
kmeans = KMeans(n_clusters=2)
kmeans.fit(X_imputed)

# Отримання міток кластерів для кожного об'єкта
labels = kmeans.labels_

# Додавання міток кластерів до даних
data['cluster'] = labels

# Вивід результів
print(data.head())

# Відображення графіка
plt.figure(figsize=(10, 6))

# Перевірка чи є точки для кожного кластера
for cluster_num in range(2):
    if data[data['cluster'] == cluster_num].shape[0] > 0:
        cluster_data = data[data['cluster'] == cluster_num]
        plt.scatter(cluster_data['odometer'], cluster_data['year'], label=f'Cluster {cluster_num}', alpha=0.5)

plt.xlabel('Пробіг')
plt.ylabel('Рік')
plt.title('Кластеризація за пробігом та роком')
plt.legend()
plt.show()

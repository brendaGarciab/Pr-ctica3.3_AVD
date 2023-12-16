"""
Created on Fri Dec 15 19:19:13 2023

@author: Bren Guzmán, Brenda García, María José Merino
"""

import seaborn as sns
import pandas as pd
import numpy as np
from corr import calcular_correlaciones, matriz_correlacion
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

#%% Cargar el conjunto de datos
data = pd.read_csv('penguins_size.csv')
data = data.loc[data['sex'].isin(['MALE', 'FEMALE'])]
# Cambiar valores nulos e infinitos
data = data.replace([np.inf, -np.inf], np.nan)
data = data.fillna(data.mean())

# Codificar la columna 'species'
data['species'] = data['species'].astype('category')
data['species_encoded'] = data['species'].cat.codes

# Codificar la columna 'sex'
data['sex'] = data['sex'].astype('category')
data['sex_encoded'] = data['sex'].cat.codes

# Codificar la columna 'island'
data['island'] = data['island'].astype('category')
data['island_encoded'] = data['island'].cat.codes

data = data.reset_index(drop=True)


#%% Crear un pairplot con Seaborn

columnas_numericas = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex_encoded', 'island_encoded', 'species_encoded']
sns.pairplot(data, hue='species', vars=columnas_numericas, corner=True)


#%% reporte de correlaciones
# Seleccionar solo las columnas numéricas 
columnas_numericas = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex_encoded', 'island_encoded']

# Calcular correlaciones y mostrar en el reporte
df_correlaciones = calcular_correlaciones(data, columnas_numericas)
print(f"\nTabla de Correlaciones:\n{df_correlaciones}")

#%% Matriz de correlación (Mapa de calor): Spearman

columnas_numericas = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex_encoded', 'island_encoded', 'species_encoded']

# Calcular la matriz de correlación

correlation_matrix = matriz_correlacion(data[columnas_numericas], metodo='spearman')

# Crear un mapa de calor
plt.figure(figsize=(10, 8))
heatmap = plt.pcolor(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)

# Anotar los valores de correlación en los cuadros
for i in range(len(correlation_matrix.index)):
    for j in range(len(correlation_matrix.columns)):
        plt.text(j + 0.5, i + 0.5, f'{correlation_matrix.iloc[i, j]:.2f}', ha='center', va='center', color='black')

plt.title('Matriz de correlación (Mapa de calor): Spearman')
plt.xticks(np.arange(0.5, len(correlation_matrix.columns), 1), correlation_matrix.columns, rotation=45)
plt.yticks(np.arange(0.5, len(correlation_matrix.index), 1), correlation_matrix.index)
plt.colorbar(heatmap)

plt.show()

#%% Matriz de correlación (Mapa de calor): Kendall

columnas_numericas = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex_encoded', 'island_encoded', 'species_encoded']

# Calcular la matriz de correlación

correlation_matrix = matriz_correlacion(data[columnas_numericas], metodo='kendall')
#correlation_matrix = data[columnas_numericas].corr(method='kendall')

# Crear un mapa de calor
plt.figure(figsize=(10, 8))
heatmap = plt.pcolor(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)

# Anotar los valores de correlación en los cuadros
for i in range(len(correlation_matrix.index)):
    for j in range(len(correlation_matrix.columns)):
        plt.text(j + 0.5, i + 0.5, f'{correlation_matrix.iloc[i, j]:.2f}', ha='center', va='center', color='black')

plt.title('Matriz de correlación (Mapa de calor): Kendall')
plt.xticks(np.arange(0.5, len(correlation_matrix.columns), 1), correlation_matrix.columns, rotation=45)
plt.yticks(np.arange(0.5, len(correlation_matrix.index), 1), correlation_matrix.index)
plt.colorbar(heatmap)

plt.show()

#%% Clasificación (Random Forest)

cols = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm']


# Dividir los datos en características (X) y variable objetivo (y)
X = data[cols]  
y = data['species_encoded']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

# Inicializar el clasificador de Bosques Aleatorios
clf = RandomForestClassifier(random_state=1998)

# Entrenar el modelo
clf.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = clf.predict(X_test)

# Evaluar el rendimiento del modelo
accuracy = accuracy_score(y_test, y_pred)
print("\n\nPrecisión del modelo:", accuracy)

# Matriz de Confusión
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nMatriz de Confusión:\n", conf_matrix)

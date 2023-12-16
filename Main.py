"""
Created on Fri Dec 15 19:19:13 2023

@author: Bren Guzmán, Brenda García, María José Merino
"""

import seaborn as sns
import pandas as pd
import numpy as np
from corr import calcular_correlaciones, matriz_correlacion
import matplotlib.pyplot as plt

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


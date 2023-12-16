"""
Created on Fri Dec 15 20:00:19 2023

@author: Bren Guzmán, Brenda García, María José Merino
"""
import pandas as pd
from scipy.stats import pearsonr
import numpy as np

def calcular_correlacion_spearman(x, y):
    n = len(x)
    ranked_x = [sorted(x).index(i) + 1 for i in x]
    ranked_y = [sorted(y).index(j) + 1 for j in y]

    d_squared = sum([(rx - ry)**2 for rx, ry in zip(ranked_x, ranked_y)])

    rho = 1 - (6 * d_squared) / (n * (n**2 - 1))
    return rho

def calcular_correlacion_kendall(x, y):
    n = len(x)
    concordant, discordant = 0, 0

    for i in range(n):
        for j in range(i + 1, n):
            if (x[i] - x[j]) * (y[i] - y[j]) > 0:
                concordant += 1
            elif (x[i] - x[j]) * (y[i] - y[j]) < 0:
                discordant += 1

    tau = (concordant - discordant) / np.sqrt((concordant + discordant) * (n * (n - 1) / 2))
    return tau

def calcular_correlaciones(data, columnas):
    resultados = []
    for columna in columnas:
        # Calcular los coeficientes de correlación
        pearson_corr, _ = pearsonr(data[columna], data['species'].astype('category').cat.codes)
        spearman_corr = calcular_correlacion_spearman(data[columna], data['species'].astype('category').cat.codes)
        kendall_corr = calcular_correlacion_kendall(data[columna], data['species'].astype('category').cat.codes)
        
        # Guardar los resultados en una lista
        resultados.append([columna, pearson_corr, spearman_corr, kendall_corr])
    
    # Crear un DataFrame con los resultados
    df_resultados = pd.DataFrame(resultados, columns=['Atributo', 'Pearson', 'Spearman', 'Kendall'])
    
    return df_resultados


def matriz_correlacion(df, metodo='pearson'):
    if metodo == 'pearson':
        return df.corr(method='pearson')
    elif metodo == 'spearman':
        return df.apply(lambda x: df.apply(lambda y: calcular_correlacion_spearman(x, y)), axis=0)
    elif metodo == 'kendall':
        return df.apply(lambda x: df.apply(lambda y: calcular_correlacion_kendall(x, y)), axis=0)
    else:
        raise ValueError("Método de correlación no válido. Use 'pearson', 'spearman' o 'kendall'.")


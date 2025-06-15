# Importamos la librería pandas y seaborn
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
# Abrimos el archivo usando una función específica de pandas
oslo_dataset = pd.read_csv("datos/temperature_Oslo_celsius.csv")
quito_dataset = pd.read_csv("datos/temperature_Quito_celsius.csv")
melbourne_dataset = pd.read_csv("datos/temperature_Melbourne_celsius.csv")

#---------------------------------------------Limpieza de datos---------------------------------------------------
"""
# Función para detectar valores atípicos utilizando como criterio el rango intercuartil
# Esta función recibe un dataset y devuelve los valores atípicos, Q1, Q3 e IQR en ese orden
"""
def analizar_rango_intercuartil(dataset):
    q1 = dataset['AvgTemperature'].quantile(0.25)
    q3 = dataset['AvgTemperature'].quantile(0.75)
    iqr = q3 - q1
    atipicos = dataset[(dataset['AvgTemperature'] < (q1 - 1.5 * iqr)) | (dataset['AvgTemperature'] > (q3 + 1.5 * iqr))]
    print(f"Q1: {q1}, Q3: {q3}, IQR: {iqr}")
    print(f"Valores atípicos: {atipicos['AvgTemperature'].values}")
    return atipicos, q1, q3, iqr

print("Datos de Oslo:")
# Utilizamos la función para analizar el rango intercuartil de Oslo
atipicos_oslo, q1_oslo, q3_oslo, iqr_oslo = analizar_rango_intercuartil(oslo_dataset)
# Vemos como existen valores fuera del rango intercuartil pero que no son outliers, como -21,-22,-23. Pero valores como -73 son imposibles
# Procedemos a eliminar los valores atípicos del dataset de Oslo
oslo_dataset = oslo_dataset[(oslo_dataset['AvgTemperature'] > -40)]

print("\nDatos de Quito:")
# Repetimos el proceso para Quito
atipicos_quito, q1_quito, q3_quito, iqr_quito = analizar_rango_intercuartil(quito_dataset)
# Existen una gran cantidad de posibles valores atípicos en Quito, pero no lo son. Esto se debe a la poca variación de la temperatura en esta ciudad

print("\nDatos de Melbourne:")
# Repetimos el proceso para Melbourne
atipicos_melbourne, q1_melbourne, q3_melbourne, iqr_melbourne = analizar_rango_intercuartil(melbourne_dataset)
# Igual que en Oslo, existen valores muy fuera del rango con valor -73, procedemos a eliminarlos
melbourne_dataset = melbourne_dataset[(melbourne_dataset['AvgTemperature'] > -40)]


#-----------------------------------------------------------------------PARTE 1--------------------------------------------------------------------

# Creamos una funcion para hacer la media recorriendo el dataset y dividiendo la suma de los valores entre el número total de estos
def calcular_media(dataset):
    suma = 0
    for valor in dataset['AvgTemperature']:
        suma += valor
    return suma / len(dataset['AvgTemperature'])

# Creamos una funcion para hacer el desvio estandar recorriendo el dataset y calculando la raiz cuadrada de la varianza
def calcular_desvio_estandar(dataset,media):
    suma_cuadrados = 0
    for valor in dataset['AvgTemperature']:
        suma_cuadrados += (valor - media) ** 2
    return (suma_cuadrados / len(dataset['AvgTemperature'])) ** 0.5


# Crear diccionario con lo pedido
media_desvio_ciudades = {
    'Ciudad': ['Oslo', 'Quito', 'Melbourne'],
    'Media': [
        calcular_media(oslo_dataset),
        calcular_media(quito_dataset),
        calcular_media(melbourne_dataset)
    ]
}
media_desvio_ciudades['Desvio_Estandar'] = [
    calcular_desvio_estandar(oslo_dataset, media_desvio_ciudades['Media'][0]),
    calcular_desvio_estandar(quito_dataset, media_desvio_ciudades['Media'][1]),
    calcular_desvio_estandar(melbourne_dataset, media_desvio_ciudades['Media'][2])
]
media_desvio_df = pd.DataFrame(media_desvio_ciudades)  
# Creamos una tabla en la que se muestra la media de la temperatura y el desvio estandar de cada ciudad
print(media_desvio_df)

#Para complementar estos datos, los analizaremos junto a la distribución de las temperaturas de cada ciudad.
#Realizamos el histograma de las temperaturas de cada ciudad para observar la distribución de los datos.

def crear_histograma(dataset, ciudad,rangox):
    plt.figure(figsize=(15, 5))
    plt.hist(dataset['AvgTemperature'], bins=30, alpha=0.7, label=ciudad,edgecolor='black')
    plt.title(f'Distribución de Temperaturas en {ciudad}')
    plt.xlabel('Temperatura (°C)')
    plt.ylabel('Frecuencia')
    #Agregamos la media y el desvío estándar al gráfico
    media = media_desvio_df[media_desvio_df['Ciudad'] == ciudad]['Media'].values[0]
    desvio_estandar = media_desvio_df[media_desvio_df['Ciudad'] == ciudad]['Desvio_Estandar'].values[0]
    plt.axvline(media, color='red', linestyle='dashed', linewidth=1, label=f'Media: {media:.2f}°C')
    plt.axvline(media + desvio_estandar, color='green', linestyle='dashed', linewidth=1, label=f'+1 Desvío Estándar: {media + desvio_estandar:.2f}°C')
    plt.axvline(media - desvio_estandar, color='green', linestyle='dashed', linewidth=1, label=f'-1 Desvío Estándar: {media - desvio_estandar:.2f}°C')
    plt.xticks(rangox)  
    
    plt.legend()
    plt.show()

# Configuración de los gráficos 


# Creamos los histogramas para cada ciudad
#crear_histograma(oslo_dataset, 'Oslo',range(-25,25,5))
#crear_histograma(quito_dataset, 'Quito',range(10,20,2))
#crear_histograma(melbourne_dataset, 'Melbourne',range(5,30,5))

#Grafico que muestra cada valor con su indice en el dataset
def crear_grafico_valores(dataset, ciudad):
    plt.figure(figsize=(15, 5))
    plt.plot(dataset['AvgTemperature'], marker='o', linestyle='-', label=ciudad)
    plt.title(f'Temperaturas en {ciudad} a lo largo del tiempo')
    plt.xlabel('Dia')
    plt.ylabel('Temperatura (°C)')
    plt.axhline(media_desvio_df[media_desvio_df['Ciudad'] == ciudad]['Media'].values[0], color='red', linestyle='dashed', linewidth=1, label='Media')
    plt.legend()
    plt.show()

crear_grafico_valores(oslo_dataset, 'Oslo')
crear_grafico_valores(quito_dataset, 'Quito')
crear_grafico_valores(melbourne_dataset, 'Melbourne')


#Quito tiene la temperatura media más estable (desvío estándar más bajo: ~1.3 °C), lo cual es típico de una ciudad ecuatorial con clima templado todo el año.
#Oslo muestra la mayor variabilidad térmica (desvío estándar de ~8.79 °C) por lo que tiene un clima más cambiante.
#Melbourne tiene la media más alta (~17.8 °C), pero también muestra una variabilidad moderada (desvío estándar de ~4.25 °C), lo que sugiere un clima templado oceánico con cambios frecuentes.

#Funcion que calcula el factor de correlacion cruzado entre dos datasets 
def calcular_correlacion_cruzada(dataset1, dataset2,nombre_ciudad1, nombre_ciudad2):
    if len(dataset1) != len(dataset2):
        raise ValueError("Los datasets deben tener la misma longitud")
    
    media1 = media_desvio_ciudades[media_desvio_df['Ciudad'] == nombre_ciudad1]['Media'].values[0]
    media2 = media_desvio_ciudades[media_desvio_df['Ciudad'] == nombre_ciudad2]['Media'].values[0]
    
    numerador = sum((dataset1['AvgTemperature'][i] - media1) * (dataset2['AvgTemperature'][i] - media2) for i in range(len(dataset1)))
    denominador = (sum((dataset1['AvgTemperature'][i] - media1) ** 2 for i in range(len(dataset1))) * 
                   sum((dataset2['AvgTemperature'][i] - media2) ** 2 for i in range(len(dataset2)))) ** 0.5
    
    if denominador == 0:
        return 0
    
    return numerador / denominador

print("\nCorrelación cruzada entre Oslo y Quito:", calcular_correlacion_cruzada(oslo_dataset, quito_dataset, 'Oslo', 'Quito'))
print("Correlación cruzada entre Oslo y Melbourne:", calcular_correlacion_cruzada(oslo_dataset, melbourne_dataset, 'Oslo', 'Melbourne'))
print("Correlación cruzada entre Quito y Melbourne:", calcular_correlacion_cruzada(quito_dataset, melbourne_dataset, 'Quito', 'Melbourne'))
#SEGUIR DESPUES
##QUE HACEMOS CON LOS DATASETS Q SOPN DE TAMANIO DIFERENTE?
#-----------------------------------------------------------------------PARTE 2--------------------------------------------------------------------
funcion_discretizar = lambda x: 'Bajo' if x < 11 else ('Medio' if 11 <= x < 19 else 'Alto')
oslo_dataset['discretizado'] = oslo_dataset['AvgTemperature'].apply(funcion_discretizar)
quito_dataset['discretizado'] = quito_dataset['AvgTemperature'].apply(funcion_discretizar)
melbourne_dataset['discretizado'] = melbourne_dataset['AvgTemperature'].apply(funcion_discretizar)

def calcular_matriz_conjunta(dataset1):
# Importamos la librería pandas
import pandas as pd

# abrimos el archivo usando una función específica de pandas
oslo_dataset = pd.read_csv("datos/temperature_Oslo_celsius.csv")
quito_dataset = pd.read_csv("datos/temperature_Quito_celsius.csv")
melbourne_dataset = pd.read_csv("datos/temperature_Melbourne_celsius.csv")

#---------------------------------------------Limpieza de datos---------------------------------------------------
print("Datos de Oslo:")
print(oslo_dataset.describe())

#Utilizaremos el Rango intercuartil para detectar y eliminar los valores atípicos
q1_oslo = oslo_dataset['AvgTemperature'].quantile(0.25)
q3_oslo = oslo_dataset['AvgTemperature'].quantile(0.75)
iqr_oslo = q3_oslo - q1_oslo
print(f"Q1: {q1_oslo}, Q3: {q3_oslo}, IQR: {iqr_oslo}")
# Filtramos los datos para eliminar los valores atípicos
atipicos=oslo_dataset[(oslo_dataset['AvgTemperature'] < (q1_oslo - 1.5 * iqr_oslo)) | (oslo_dataset['AvgTemperature'] > (q3_oslo + 1.5 * iqr_oslo))]
print(f"Valores atípicos en Oslo: {atipicos['AvgTemperature'].values}")
#Aca vemos como existen valores fuera del rango intercuartil pero que no son outliers, como -21,-22,-23. Pero valores como -73 son imposibles

# Eliminamos los valores atípicos del dataset de Oslo
oslo_dataset = oslo_dataset[(oslo_dataset['AvgTemperature'] > -40)]





#-----------------------------------------------------------------------PARTE 1--------------------------------------------------------------------

# creamos una funcion para hacer la media recorriendo el dataset y dividiendo la suma de los valores entre el número de valores
def calcular_media(dataset):
    suma = 0
    for valor in dataset['AvgTemperature']:
        suma += valor
    return suma / len(dataset['AvgTemperature'])

#creamos una funcion para hacer el desvio estandar recorriendo el dataset y calculando la raiz cuadrada de la varianza
def calcular_desvio_estandar(dataset,media):
    suma_cuadrados = 0
    for valor in dataset['AvgTemperature']:
        suma_cuadrados += (valor - media) ** 2
    return (suma_cuadrados / len(dataset['AvgTemperature'])) ** 0.5


# Crear diccionario con lo pedido
resumen = {
    'Ciudad': ['Oslo', 'Quito', 'Melbourne'],
    'Media': [
        calcular_media(oslo_dataset),
        calcular_media(quito_dataset),
        calcular_media(melbourne_dataset)
    ]
}
resumen['Desvio_Estandar'] = [
    calcular_desvio_estandar(oslo_dataset, resumen['Media'][0]),
    calcular_desvio_estandar(quito_dataset, resumen['Media'][1]),
    calcular_desvio_estandar(melbourne_dataset, resumen['Media'][2])
]
resumen_df = pd.DataFrame(resumen)  
# Creamos una tabla en la que se muetra la media de la temperatura y el desvio estandar de cada ciudad
print(resumen_df)
oslo_dataset.describe()


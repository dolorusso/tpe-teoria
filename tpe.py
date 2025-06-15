# Importamos la librería pandas y seaborn
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# Configuracion de librerias
#Mostramos 3 decimales, no usamos notacion cientifica y mostramos los numeros flotantes enteros(esto es para que cuando sean 0 se vean igual)
np.set_printoptions(precision=3, suppress=True, floatmode='fixed')


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

#F (frío): si t < 11°C
#T (templado): si 11 ≤ t < 19°C
#C (cálido): si t ≥ 19°C
#Definir la función de discretización
funcion_disc = lambda t: 'F' if t < 11 else 'T' if t < 19 else 'C'

# Aplicar la función de discretización a la columna 'AvgTemperature' de cada dataset, creando una nueva columna 'clima'
oslo_dataset['clima'] = oslo_dataset['AvgTemperature'].apply(funcion_disc)
quito_dataset['clima'] = quito_dataset['AvgTemperature'].apply(funcion_disc)
melbourne_dataset['clima'] = melbourne_dataset['AvgTemperature'].apply(funcion_disc)

#matriz de probabilidad conjunta
def calcular_matriz_conjunta(df):

    # Obtener los estados únicos
    states = ['F', 'T', 'C']
    
    # Crear un diccionario para mapear estados a índices, asi puedo acceder a la matriz con indices numericos
    estado_indice = {state: i for i, state in enumerate(states)}
    
    # Inicializar matriz de conteos
    matriz_conjunta = np.zeros((3, 3))
    
    # Contar los eventos
    cantidad_eventos = len(df) - 1 #-1
    for i in range(cantidad_eventos):
        x = df['clima'].iloc[i]
        y = df['clima'].iloc[i + 1]
        matriz_conjunta[estado_indice[x], estado_indice[y]] += 1

    # Calcular la matriz conjunta dividiendo la cantidad de eventos por el total de eventos, precision 3 decimales
    matriz_conjunta = matriz_conjunta / cantidad_eventos if cantidad_eventos > 0 else matriz_conjunta

    '''# Convertir a DataFrame
    matriz_df = pd.DataFrame(matriz_conjunta, index=states, columns=states)'''

    return matriz_conjunta

# Calcular la matriz de probabilidad conjunta para cada dataset, ya que esta matriz es la que utilizaremos para calcular otras
matriz_conjunta_oslo = calcular_matriz_conjunta(oslo_dataset)
matriz_conjunta_quito = calcular_matriz_conjunta(quito_dataset)
matriz_conjunta_melbourne = calcular_matriz_conjunta(melbourne_dataset)

print("Matriz de probabilidad conjunta de Oslo:\n" , matriz_conjunta_oslo)
print("Matriz de probabilidad conjunta de Quito:\n" , matriz_conjunta_quito)
print("Matriz de probabilidad conjunta de Melbourne:\n" , matriz_conjunta_melbourne)

print("verificar suma da 1: ", np.sum(matriz_conjunta_oslo) == 1)
print("verificar suma da 1: ", np.sum(matriz_conjunta_quito) == 1)
print("verificar suma da 1: ", np.sum(matriz_conjunta_melbourne) == 1)

# Función para calcular la probabilidad de transición a partir de la matriz conjunta
def calcular_matriz_transicion(matriz_conjunta):
    #Obtenemos la probilidad de cada columna
    prob_marginal_x = [0] * 3
    for i in range(3):
        prob_marginal_x[i] = matriz_conjunta[i,0] + matriz_conjunta[i,1] + matriz_conjunta[i,2]#harcodeado, importa?

    # Inicializar matriz de transición
    matriz_transicion = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            # Calcular la probabilidad de transición
            matriz_transicion[i, j] = matriz_conjunta[i, j] / prob_marginal_x[j] if prob_marginal_x[j] > 0 else 0
    return matriz_transicion


# Calculamos la matriz de transicion para cada dataset
matriz_transicion_oslo = calcular_matriz_transicion(matriz_conjunta_oslo)
matriz_transicion_quito = calcular_matriz_transicion(matriz_conjunta_quito)
matriz_transicion_melbourne = calcular_matriz_transicion(matriz_conjunta_melbourne)


# Imprimimos las matrices de transicion
print("Matriz de transición de Oslo:\n", matriz_transicion_oslo)
print("Matriz de transición de Quito:\n", matriz_transicion_quito)
print("Matriz de transición de Melbourne:\n", matriz_transicion_melbourne)

#A partir de esta matriz de transicion, calcularemos el vector estacionario con motor de monte carlo
# Definimos una funcion que calcula la matriz acumulada de cada fuente
def calcular_matriz_acumulada(matriz_transicion):
    matriz_resultado = np.zeros((3, 3))  # Inicializamos la matriz de resultado
    matriz_resultado = matriz_transicion.copy()  # Copiamos la primer fila
    for i in range(1,3):
        for j in range(0,3):
            matriz_resultado[i, j] += matriz_resultado[i-1, j]
    return matriz_resultado

# Definimos la matriz acumulada de cada fuente
matriz_acumulada_oslo = calcular_matriz_acumulada(matriz_transicion_oslo)
matriz_acumulada_quito = calcular_matriz_acumulada(matriz_transicion_quito)
matriz_acumulada_melbourne = calcular_matriz_acumulada(matriz_transicion_melbourne)

print("Matriz acumulada de Oslo:\n", matriz_acumulada_oslo)
print("Matriz acumulada de Quito:\n", matriz_acumulada_quito)
print("Matriz acumulada de Melbourne:\n", matriz_acumulada_melbourne)

#Definimos las funciones auxilineares necesarias
def converge_vector(vector1, vector2, e=0.00001):
    # Compara dos vectores y verifica si la diferencia es menor que el umbral epsilon
    for i in range(len(vector1)):
        if abs(vector1[i] - vector2[i]) >= e:
            return False
    return True

# Genera el próximo estado basado en la matriz acumulada y el simbolo actual
def generar_proximo_estado(matriz_acumulada, simbolo_anterior):
    n_random = np.random.rand()  # Genera un número aleatorio entre 0 y 1
    for i in range(3):
        if matriz_acumulada[i, simbolo_anterior] > n_random:
            return i
    raise ValueError("No se pudo generar el próximo estado, matriz acumulada no válida.")


def calcular_vector_estacionario(matriz_acumulada, e=0.00001, min_iter=5000):
    # Inicializar el vector estacionario
    vector_estacionario = np.array([1/3, 1/3, 1/3])  # Distribución uniforme inicial
    emisiones = np.array([0, 0, 0])  # Contador de emisiones para cada estado
    Vt_actual = np.array([0, 0, 0])  # Vector de emisiones actual
    Vt_anterior = np.array([0, 0, 0])  # Vector de emisiones anterior
    cantidad_simbolos = 0  # Contador de símbolos generados
    simbolo_actual = 0  # Estado actual, no importa su valor inicial al calcular el vector estacionario

    while not converge_vector(vector_estacionario, Vt_anterior, e) or cantidad_simbolos < min_iter:
        # Generamos el proximo simbolo
        simbolo_actual = generar_proximo_estado(matriz_acumulada, simbolo_actual)
        emisiones[simbolo_actual] += 1
        cantidad_simbolos += 1
        # Actualizamos el vector de emisiones
        Vt_anterior = Vt_actual.copy()
        Vt_actual = emisiones / cantidad_simbolos
    
    return Vt_actual

# Calculamos el vector estacionario para cada dataset
vector_estacionario_oslo = calcular_vector_estacionario(matriz_acumulada_oslo)
vector_estacionario_quito = calcular_vector_estacionario(matriz_acumulada_quito)
vector_estacionario_melbourne = calcular_vector_estacionario(matriz_acumulada_melbourne)

# Imprimimos los vectores estacionarios
print("Vector estacionario de Oslo:\n", vector_estacionario_oslo)
print("Vector estacionario de Quito:\n", vector_estacionario_quito)
print("Vector estacionario de Melbourne:\n", vector_estacionario_melbourne)

#Comprobacion de salida
print("Comprobacion Vector estacionario de Oslo:\n", round(sum(vector_estacionario_oslo), 3))
print("Comprobacion Vector estacionario de Quito:\n", round(sum(vector_estacionario_quito), 3))
print("Comprobacion Vector estacionario de Melbourne:\n", round(sum(vector_estacionario_melbourne), 3))

#Definimos las funciones auxilineares necesarias
def converge(media_actual, media_anterior, e=0.00001):
    return abs(media_actual - media_anterior) < e

# Funcion para calcular el tiempo medio de primera recurrencia
def calcular_tiempo_recurrencia(matriz_acumulada, estado_inicial, e=0.00001, min_iter=5000):
    retornos = 0
    media_actual = 0
    media_anterior = 0
    cantidad_iteraciones = 0
    simbolo_actual = estado_inicial
    while not converge(media_actual, media_anterior, e) or cantidad_iteraciones < min_iter:
        # Generamos el proximo simbolo
        simbolo_actual = generar_proximo_estado(matriz_acumulada, simbolo_actual)
        cantidad_iteraciones += 1
        if simbolo_actual == estado_inicial:
            retornos += 1
            media_anterior = media_actual
            media_actual = cantidad_iteraciones / retornos
    return media_actual

# Calculamos el tiempo medio de primera recurrencia para cada dataset, para cada estado y guardamos en un dataframe
def calcular_tiempos_recurrencia(matriz_acumulada, ciudad):
    estados = ['F', 'T', 'C']
    tiempos_recurrencia = {}
    for i, estado in enumerate(estados):
        tiempo = calcular_tiempo_recurrencia(matriz_acumulada, i)
        tiempos_recurrencia[estado] = tiempo
    return pd.DataFrame([tiempos_recurrencia], index=[ciudad])

# Calculamos los tiempos de recurrencia para cada dataset
tiempos_recurrencia_oslo = calcular_tiempos_recurrencia(matriz_acumulada_oslo, 'Oslo')
tiempos_recurrencia_quito = calcular_tiempos_recurrencia(matriz_acumulada_quito, 'Quito')
tiempos_recurrencia_melbourne = calcular_tiempos_recurrencia(matriz_acumulada_melbourne, 'Melbourne')

#Juntamos los dataframes de tiempos de recurrencia en un frame e imprimimos el resultado
tiempos_recurrencia = pd.concat([tiempos_recurrencia_oslo, tiempos_recurrencia_quito, tiempos_recurrencia_melbourne])   
print("Tiempos de recurrencia:\n", tiempos_recurrencia)

# --------------------------------- PARTE 3 ----------------------------------------------------#
# Definimos una funcion para calcular la entropia de orden 0 de cada dataset
def calcular_entropia_orden_0(vector_estacionario):
    # La entropia de orden 0 se calcula como -sum(p * log(p)) para cada estado
    entropia = 0
    for i in range(len(vector_estacionario)):
        if vector_estacionario[i] > 0:  # Evitar log(0) pq da error
            entropia -= vector_estacionario[i] * np.log2(vector_estacionario[i])
    return entropia

# Definimos una funcion para calcular la entropia de orden 1 de cada dataset
def calcular_entropia_orden_1(matriz_conjunta):
    # La entropia de orden 1 se calcula como -sum(p(x,y) * log(p(x,y))) para cada par de estados
    entropia = 0
    for i in range(3):
        for j in range(3):
            if matriz_conjunta[i, j] > 0:
                entropia -= matriz_conjunta[i, j] * np.log2(matriz_conjunta[i, j])
    return entropia

# Calculamos la entropia de orden 0 y 1 para cada dataset
H0_oslo = calcular_entropia_orden_0(vector_estacionario_oslo)
H0_quito = calcular_entropia_orden_0(vector_estacionario_quito)
H0_melbourne = calcular_entropia_orden_0(vector_estacionario_melbourne)
H1_oslo = calcular_entropia_orden_1(matriz_conjunta_oslo)
H1_quito = calcular_entropia_orden_1(matriz_conjunta_quito)
H1_melbourne = calcular_entropia_orden_1(matriz_conjunta_melbourne)
# Imprimimos las entropias de orden 0 y 1 en un dataframe conjunto
entropias = pd.DataFrame({
    'H0': [H0_oslo, H0_quito, H0_melbourne],
    'H1': [H1_oslo, H1_quito, H1_melbourne]
}, index=['Oslo', 'Quito', 'Melbourne'])
print("Entropías de orden 0 y 1:\n", entropias)

#Implementamos una funcion que nos de codigo de Huffman

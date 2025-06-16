# Importamos la librería pandas y seaborn
#import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import heapq
# Configuracion de librerias
#Mostramos 3 decimales, no usamos notacion cientifica y mostramos los numeros flotantes enteros(esto es para que cuando sean 0 se vean igual)
np.set_printoptions(precision=3, suppress=True, floatmode='fixed')


# Abrimos el archivo usando una función específica de pandas
oslo_dataset = pd.read_csv("temperature_Oslo_celsius.csv")
quito_dataset = pd.read_csv("temperature_Quito_celsius.csv")
melbourne_dataset = pd.read_csv("temperature_Melbourne_celsius.csv")
melbourne_ruidoso_dataset = pd.read_csv('temperature_Melbourne_celsius_ruidoso.csv')


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
#Antes guardamos el indice de los valores atípicos para poder eliminarlos en segundo dataset de melbourne
indices_atipicos_melbourne = atipicos_melbourne.index
melbourne_dataset = melbourne_dataset[(melbourne_dataset['AvgTemperature'] > -40)]
melbourne_ruidoso_dataset = melbourne_ruidoso_dataset.drop(indices_atipicos_melbourne, axis=0)

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
"""
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
"""
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

#----------------------------------------------------CALCULO DEL VECTOR ESTACIONARIO----------------------------------------------------

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


def calcular_vector_estacionario(matriz_acumulada, e=0.0001, min_iter=5000):
    # Inicializar el vector estacionario
    emisiones = np.array([0, 0, 0])  # Contador de emisiones para cada estado
    Vt_actual = np.array([0, 0, 0])  # Vector de emisiones actual
    Vt_anterior = np.array([0, 0, 0])  # Vector de emisiones anterior
    cantidad_simbolos = 0  # Contador de símbolos generados
    simbolo_actual = 0  # Estado actual, no importa su valor inicial al calcular el vector estacionario

    while not converge_vector(Vt_actual, Vt_anterior, e) or cantidad_simbolos < min_iter:
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


#--------------------------------------------------------------CALCULO DEL TIEMPO MEDIO DE PRIMERA RECURRENCIA----------------------------------------------------
#Definimos las funciones auxilineares necesarias
def converge(media_actual, media_anterior, e=0.00001):
    return abs(media_actual - media_anterior) < e

# Funcion para calcular el tiempo medio de primera recurrencia
def calcular_tiempo_recurrencia(matriz_acumulada, estado_inicial, e=0.001, min_iter=5000):
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

"""# Definimos una funcion para calcular la entropia condicional
def calcular_entropia_condicional(matriz_conjunta):
    # La entropia de orden 1 se calcula como -sum(p(x,y) * log(p(x,y))) para cada par de estados
    entropia = 0
    for i in range(3):
        for j in range(3):
            if matriz_conjunta[i, j] > 0:
                entropia -= matriz_conjunta[i, j] * np.log2(matriz_conjunta[i, j])
    return entropia"""
    
def calcular_entropia_condicional(matriz_transicion, vector_estacionario):
    entropia_condicional = 0
    for i in range(3):
        h_i = 0
        for j in range(3):
            if matriz_transicion[i, j] > 0:
                h_i -= matriz_transicion[i, j] * np.log2(matriz_transicion[i, j])
        entropia_condicional += vector_estacionario[i] * h_i
    return entropia_condicional

# Calculamos la entropia de orden 0 y 1 para cada dataset
H0_oslo = calcular_entropia_orden_0(vector_estacionario_oslo)
H0_quito = calcular_entropia_orden_0(vector_estacionario_quito)
H0_melbourne = calcular_entropia_orden_0(vector_estacionario_melbourne)

Hcond_oslo = calcular_entropia_condicional(matriz_transicion_oslo, vector_estacionario_oslo)
Hcond_quito = calcular_entropia_condicional(matriz_transicion_quito, vector_estacionario_quito)
Hcond_melbourne = calcular_entropia_condicional(matriz_transicion_melbourne, vector_estacionario_melbourne)
# Imprimimos las entropias de orden 0 y 1 en un dataframe conjunto
entropias = pd.DataFrame({
    'H0': [H0_oslo, H0_quito, H0_melbourne],
    'Hcond': [Hcond_oslo, Hcond_quito, Hcond_melbourne]
}, index=['Oslo', 'Quito', 'Melbourne'])
print("Entropías H1 y Hcond:\n", entropias)

#Implementamos una funcion que nos de codigo de Huffman

class NodoHuffmann:
    def __init__(self, probabilidad, simbolo=None, izq=None, der=None):
        self.simbolo = simbolo
        self.probabilidad = probabilidad
        self.izq = izq
        self.der = der

    def __lt__(self, otro): #Creamos esto para comparar nodos mas facil y que se pueda utilizar en la cola de prioridad
        return self.probabilidad < otro.probabilidad
    

def calcular_huffman(probabilidades):
    n = len(probabilidades)
    
    #Creamos el minheap para almacenar los nodos y acceder de manera eficiente a los menor probabilidad
    codequeue = []
    for simbolo, probabilidad in probabilidades.items():
        if probabilidad == 0:
            continue
        nodo = NodoHuffmann(probabilidad, simbolo)
        heapq.heappush(codequeue, nodo)
    while len(codequeue) > 1:
        #Sacamos los 2 nodos con menor probabilidad
        menor_probabilidad = heapq.heappop(codequeue) 
        segundo_menor_probabilidad = heapq.heappop(codequeue)
        #los combinamos en un nuevo nodo padre
        nodo_padre = NodoHuffmann(
            probabilidad=menor_probabilidad.probabilidad + segundo_menor_probabilidad.probabilidad,
            izq=menor_probabilidad,
            der=segundo_menor_probabilidad
        )
        heapq.heappush(codequeue, nodo_padre)
        
    # El ultimo nodo de la cola es la raíz del árbol de Huffman
    raiz = codequeue[0]
    # Generamos los códigos de Huffman
    codigos = {}
    def generar_codigos_huffman(nodo, codigo=''):
        if nodo.simbolo is not None:
            codigos[nodo.simbolo] = codigo
        else:
            generar_codigos_huffman(nodo.izq, codigo + '0')
            generar_codigos_huffman(nodo.der, codigo + '1')
    
    
    generar_codigos_huffman(raiz)
    #Pasamos el diccionario a un DataFrame
    codigos = pd.DataFrame(list(codigos.items()), columns=['Simbolo', 'Codigo'])
    return codigos

#llamamos a la funcion de codificacion de Huffman para cada dataset
codigos_huffman_oslo = calcular_huffman({
    'F': vector_estacionario_oslo[0],
    'T': vector_estacionario_oslo[1],
    'C': vector_estacionario_oslo[2]
})
print("Códigos de Huffman para Oslo:", codigos_huffman_oslo)

#Calculamos para quito
codigos_huffman_quito = calcular_huffman({
    'F': vector_estacionario_quito[0],
    'T': vector_estacionario_quito[1],
    'C': vector_estacionario_quito[2]
})
print("Códigos de Huffman para Quito:", codigos_huffman_quito)

#Calculamos para melbourne
codigos_huffman_melbourne = calcular_huffman({
    'F': vector_estacionario_melbourne[0],
    'T': vector_estacionario_melbourne[1],
    'C': vector_estacionario_melbourne[2]
})
print("Códigos de Huffman para Melbourne:", codigos_huffman_melbourne)

#Extendemos las fuentes a orden 2
def extender_fuente_orden_2(matriz_transicion, vector_estacionario):
    estados = ['F', 'T', 'C']
    fuente_orden_2 = {}
    
    for i in range(3):
        for j in range(3):
            estado = estados[i] + estados[j]
            probabilidad = matriz_transicion[i, j] * vector_estacionario[i]
            fuente_orden_2[estado] = probabilidad
    # Guardamos en un DF para mayor legibilidad
    fuente_orden_2 = pd.DataFrame(list(fuente_orden_2.items()), columns=['Estado', 'Probabilidad'])
    return fuente_orden_2

# Extendemos las fuentes a orden 2 para cada dataset
fuente_orden_2_oslo = extender_fuente_orden_2(matriz_transicion_oslo, vector_estacionario_oslo)
fuente_orden_2_quito = extender_fuente_orden_2(matriz_transicion_quito, vector_estacionario_quito)
fuente_orden_2_melbourne = extender_fuente_orden_2(matriz_transicion_melbourne, vector_estacionario_melbourne)

print("Fuente de orden 2 para Oslo:\n", fuente_orden_2_oslo)
print("Fuente de orden 2 para Quito:\n", fuente_orden_2_quito)
print("Fuente de orden 2 para Melboure\n:", fuente_orden_2_melbourne)

#calculamos los codigos de Huffman para las fuentes de orden 2
#primero pasamos las probabilidades de la fuente de orden 2 a un diccionario
def calcular_huffman_orden_2(fuente_orden_2):
    probabilidades = {row['Estado']: row['Probabilidad'] for _, row in fuente_orden_2.iterrows()}
    return calcular_huffman(probabilidades)

codigos_huffman_orden_2_oslo = calcular_huffman_orden_2(fuente_orden_2_oslo)
print("Códigos de Huffman de orden 2 para Oslo:\n", codigos_huffman_orden_2_oslo)

codigos_huffman_orden_2_quito = calcular_huffman_orden_2(fuente_orden_2_quito)
print("\nCódigos de Huffman de orden 2 para Quito:\n", codigos_huffman_orden_2_quito)

codigos_huffman_orden_2_melbourne = calcular_huffman_orden_2(fuente_orden_2_melbourne)
print("\nCódigos de Huffman de orden 2 para Melbourne:\n", codigos_huffman_orden_2_melbourne)

#Calculamos el limite inferior y superior de la longitud de los codigos de Huffman segun el teorema de Shannon
def calcular_limite_huffman(H1,Hcond=0,n=1):
    # H1 es la entropia de orden 1, Hcond es la entropia condicional y n es la extension de la fuente
    # Calculamos el limite inferior y superior de la longitud de los codigos de Huffman
    limite_inferior = H1/n + (1-1/n) * Hcond
    limite_superior = limite_inferior + 1/n
    return limite_inferior, limite_superior

def calcular_longitud_media(codigos_huffman, fuente):
    longitud_total = 0
    for _, row in codigos_huffman.iterrows():
        simbolo = row['Simbolo']
        codigo = row['Codigo']
        probabilidad = fuente[fuente['Estado'] == simbolo]['Probabilidad'].values[0] #Values pq devuelve una serie pandas
        longitud_total += len(codigo) * probabilidad

    return longitud_total

#Comparamos los limites de longitud de los codigos de Huffman con la longitud media por simbolo de los codigos
#Oslo
print("limite inferior y superior de la longitud de los codigos de Huffman para Oslo:")
limite_inferior_oslo, limite_superior_oslo = calcular_limite_huffman(H0_oslo, Hcond_oslo, 2)
print(f"Limite inferior: {limite_inferior_oslo:.3f}, Limite superior: {limite_superior_oslo:.3f}")
longitud_media_oslo = calcular_longitud_media(codigos_huffman_orden_2_oslo, fuente_orden_2_oslo)
print(f"Longitud media por simbolo de los codigos de Huffman para Oslo: {longitud_media_oslo/2:.3f}")

#Quito
print("\nLimite inferior y superior de la longitud de los codigos de Huffman para Quito:")
limite_inferior_quito, limite_superior_quito = calcular_limite_huffman(H0_quito, Hcond_quito, 2)
print(f"Limite inferior: {limite_inferior_quito:.3f}, Limite superior: {limite_superior_quito:.3f}")
longitud_media_quito = calcular_longitud_media(codigos_huffman_orden_2_quito, fuente_orden_2_quito)
print(f"Longitud media por simbolo de los codigos de Huffman para Quito: {longitud_media_quito/2:.3f}")

#Melbourne
print("\nLimite inferior y superior de la longitud de los codigos de Huffman para Melbourne:")
limite_inferior_melbourne, limite_superior_melbourne = calcular_limite_huffman(H0_melbourne, Hcond_melbourne, 2)
print(f"Limite inferior: {limite_inferior_melbourne:.3f}, Limite superior: {limite_superior_melbourne:.3f}")
longitud_media_melbourne = calcular_longitud_media(codigos_huffman_orden_2_melbourne, fuente_orden_2_melbourne)
print(f"Longitud media por simbolo de los codigos de Huffman para Melbourne: {longitud_media_melbourne/2:.3f}")

# Definimos la funcion que codifique las temperaturas de cada ciudad, dada una lista de simbolos, devuelve un string con el codigo total
def codificar_temperaturas(simbolos_sin_comprimir, codigos_huffman):
    codificacion = ''
    for simbolo in simbolos_sin_comprimir:
        codigo = codigos_huffman[codigos_huffman['Simbolo'] == simbolo]['Codigo'].values[0]
        codificacion += codigo
    return codificacion

def codificar_temperaturas_orden_2(simbolos_sin_comprimir, codigos_huffman):
    #Creamos una lista con los pares de simbolos no superpuestos
    dataset_pares = []
    for i in range(0,len(simbolos_sin_comprimir) - 1,2):
        # Nos aseguramos de que no haya un simbolo sobrante al final
        if i + 1 >= len(simbolos_sin_comprimir):
            break
        par = simbolos_sin_comprimir[i] + simbolos_sin_comprimir[i + 1]
        dataset_pares.append(par)
    
    # Codificamos las temperaturas usando la funcion ya definida
    codificacion = codificar_temperaturas(dataset_pares, codigos_huffman)
    return codificacion

#-------------------------------- ORDEN 1 --------------------------------#

# Codificamos las temperaturas de cada ciudad
codificacion_oslo = codificar_temperaturas(oslo_dataset['clima'].tolist(), codigos_huffman_oslo)
codificacion_quito = codificar_temperaturas(quito_dataset['clima'].tolist(), codigos_huffman_quito)
codificacion_melbourne = codificar_temperaturas(melbourne_dataset['clima'].tolist(), codigos_huffman_melbourne)

# Imprimimos las longitudes de cada codificaciones y las comparamos con el espacio que ocupara sin compresion
print("\nLongitud de la codificación de Oslo:", len(codificacion_oslo), "bits")
# Comparamos con la longitud de la codificación sin compresión, suponiendo que cada simbolo ocupa 1 byte (tamano de char)
print("La longitud de la codificación sin compresión sería:", len(oslo_dataset) * 8, "bits")
print("Por lo tanto, la compresión es de:", (len(oslo_dataset) * 8 - len(codificacion_oslo)) / (len(oslo_dataset) * 8) * 100, "%")
# Repetimos para las otras 2
print("\nLongitud de la codificación de Quito:", len(codificacion_quito), "bits")
print("La longitud de la codificación sin compresión sería:", len(quito_dataset) * 8, "bits")
print("Por lo tanto, la compresión es de:", (len(quito_dataset) * 8 - len(codificacion_quito)) / (len(quito_dataset) * 8) * 100, "%")
print("\nLongitud de la codificación de Melbourne:", len(codificacion_melbourne), "bits")
print("La longitud de la codificación sin compresión sería:", len(melbourne_dataset) * 8, "bits")  
print("Por lo tanto, la compresión es de:", (len(melbourne_dataset) * 8 - len(codificacion_melbourne)) / (len(melbourne_dataset) * 8) * 100, "%")

#-------------------------------- ORDEN 2 --------------------------------#
# Codificamos las temperaturas de cada ciudad
codificacion_oslo = codificar_temperaturas_orden_2(oslo_dataset['clima'].tolist(), codigos_huffman_orden_2_oslo)
codificacion_quito = codificar_temperaturas_orden_2(quito_dataset['clima'].tolist(), codigos_huffman_orden_2_quito)
codificacion_melbourne = codificar_temperaturas_orden_2(melbourne_dataset['clima'].tolist(), codigos_huffman_orden_2_melbourne)

#Oslo
# Imprimimos las longitudes de cada codificaciones y las comparamos con el espacio que ocupara sin compresion
print("\nLongitud de la codificación de Oslo:", len(codificacion_oslo), "bits")
# Comparamos con la longitud de la codificación sin compresión
print("La longitud de la codificación sin compresión sería:", len(oslo_dataset) * 8, "bits")  # 1B por simbolo, pq no estan juntos en el dataset
print("Por lo tanto, la compresión es de:", (len(oslo_dataset) * 8 - len(codificacion_oslo)) / (len(oslo_dataset) * 8) * 100, "%")

#Repetimos para las otras 2

#Quito
print("\nLongitud de la codificación de Quito:", len(codificacion_quito), "bits")
print("La longitud de la codificación sin compresión sería:", len(quito_dataset) * 8, "bits")  
print("Por lo tanto, la compresión es de:", (len(quito_dataset) * 8 - len(codificacion_quito)) / (len(quito_dataset) * 8) * 100, "%")

#Melbourne
print("\nLongitud de la codificación de Melbourne:", len(codificacion_melbourne), "bits")
print("La longitud de la codificación sin compresión sería:", len(melbourne_dataset) * 8, "bits")
print("Por lo tanto, la compresión es de:", (len(melbourne_dataset) * 8 - len(codificacion_melbourne)) / (len(melbourne_dataset) * 8) * 100, "%")

#---------------------------------------------------- Parte 4 ------------------------------------------------------------------------------#

#Adjuntamos el clima de entrada en la tabla, para manejarlo mas facil
melbourne_ruidoso_dataset['clima_entrada'] = melbourne_dataset['clima']

# Generamos los simbolos de las temperaturas de cada ciudad (F, T, C)
melbourne_ruidoso_dataset['clima_salida'] = melbourne_ruidoso_dataset['AvgTemperature'].apply(funcion_disc)
print("\nDatos de entrada y salida de Melbourne ruidoso:\n", melbourne_ruidoso_dataset)

# Definimos la funcion que calcule la matriz de transicion del canal, teniendo en cuenta la entrada clima y la salida clima
def calcular_matriz_transicion_canal(dataset):
    estados = ['F', 'T', 'C']
    suma_entradas = {'F': 0, 'T': 0, 'C': 0}  # Suma de las entradas para cada estado
    matriz_transicion = pd.DataFrame(0, index=estados, columns=estados, dtype=float)
    for i in range(len(dataset)):
        entrada = dataset['clima_entrada'].iloc[i]
        salida = dataset['clima_salida'].iloc[i]
        matriz_transicion.loc[entrada,salida] += 1
        suma_entradas[entrada] += 1

    # Dividimos por cada suma
    for entrada in estados:
        total = suma_entradas[entrada]
        if total > 0:
            matriz_transicion.loc[entrada] /= total
    return matriz_transicion.T #Traspuesta pq la calcule al reves con entrada,salida

#Calculamos la matriz de transicion del canal T4
matriz_transicion_t4 = calcular_matriz_transicion_canal(melbourne_ruidoso_dataset)

#Calculamos el ruido del canal
def calcular_ruido_canal(matriz_transicion,vector_estacionario):
    rx = [0, 0, 0]  # Inicializamos el vector de ruido para cada estado
    for i in range(3):
        for j in range(3):
            if matriz_transicion.iloc[i, j] > 0:
                rx[i] -= matriz_transicion.iloc[i, j] * np.log2(matriz_transicion.iloc[i, j])
    ruido = rx[0] * vector_estacionario[0] + rx[1] * vector_estacionario[1] + rx[2] * vector_estacionario[2]
    return ruido #VER QUE EL ESTACIONARIO ESTE EN ESTE ORDEN!!!!

ruido_canal = calcular_ruido_canal(matriz_transicion_t4, vector_estacionario_melbourne)
print("\nRuido del canal T4:", ruido_canal)

# Calculamos la informacion mutua del canal
def calcular_informacion_mutua(matriz_transicion, vector_estacionario):
    informacion_mutua = 0
    #Calculamos la entropia de la salida del canal
    
    return informacion_mutua

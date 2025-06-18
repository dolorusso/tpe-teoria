
def mostrar_tabla_matplotlib(media_desvio_df):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Crear la tabla
    tabla = ax.table(
        cellText=[[f"{row['Media']:.4f}", f"{row['Desvio_Estandar']:.4f}"] 
                 for _, row in media_desvio_df.iterrows()],
        rowLabels=media_desvio_df.index,
        colLabels=['Media', 'Desvío Estándar'],
        cellLoc='center',
        loc='center',
        colWidths=[0.3, 0.3]
    )
    
    # Estilo de la tabla
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(12)
    tabla.scale(1.2, 2)
    
    # Colores para encabezados
    for i in range(len(media_desvio_df.columns)):
        tabla[(0, i)].set_facecolor('#366092')
        tabla[(0, i)].set_text_props(weight='bold', color='white')
    
    # Colores para labels de filas
    for i in range(len(media_desvio_df)):
        tabla[(i+1, -1)].set_facecolor('#366092')
        tabla[(i+1, -1)].set_text_props(weight='bold', color='white')
    
    # Colores alternados para celdas
    for i in range(len(media_desvio_df)):
        for j in range(len(media_desvio_df.columns)):
            if i % 2 == 0:
                tabla[(i+1, j)].set_facecolor('#F0F8FF')
            else:
                tabla[(i+1, j)].set_facecolor('#F0F8FF')
    
    plt.title('Medidas de tendencia central', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()
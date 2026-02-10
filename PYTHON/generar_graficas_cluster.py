import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

#? Script para generar gráficas de análisis por cluster

ruta_actual = os.path.dirname(os.path.abspath(__file__))
ruta_raiz = os.path.dirname(ruta_actual)
ruta_data = os.path.join(ruta_raiz, 'DATA')
ruta_analisis = os.path.join(ruta_raiz, 'analisis')

# Crear carpeta si no existe
if not os.path.exists(ruta_analisis):
    os.makedirs(ruta_analisis)

print("="*70)
print("  GENERANDO GRÁFICAS DE ANÁLISIS POR CLUSTER")
print("="*70 + "\n")

# Cargar datos
print("Cargando datos...\n")
ratings = pd.read_csv(os.path.join(ruta_data, 'ratings_limpio.csv'))
peliculas = pd.read_csv(os.path.join(ruta_data, 'peliculas_limpio.csv'))
usuarios_features = pd.read_csv(os.path.join(ruta_data, 'usuarios_clusters.csv'))

# Configurar estilo
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (14, 6)

# GRÁFICA 1: GÉNEROS MÁS POPULARES POR CLUSTER
print("Generando gráfica 1: Géneros más populares por cluster...")

genero_cols = [col for col in peliculas.columns if col.startswith('genero_')]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Géneros Más Populares por Cluster', fontsize=16, fontweight='bold', y=1.02)

nombres_clusters = {0: "Selectivos", 1: "Críticos Activos", 2: "Cinéfilos"}

# Mapeo de géneros (basado en MovieLens)
generos_nombres = {
    0: 'Acción', 1: 'Aventura', 2: 'Animación', 3: 'Infantil', 4: 'Comedia',
    5: 'Crimen', 6: 'Documental', 7: 'Drama', 8: 'Fantasía', 9: 'Film-Noir',
    10: 'Horror', 11: 'Musical', 12: 'Misterio', 13: 'Romance', 14: 'Sci-Fi',
    15: 'Thriller', 16: 'Guerra', 17: 'Western', 18: 'IMAX'
}

for cluster_id in range(3):
    # Obtener usuarios del cluster
    usuarios_cluster = usuarios_features[usuarios_features['cluster'] == cluster_id]['user_id'].values
    
    # Películas vistas por usuarios del cluster
    pelis_cluster = ratings[ratings['user_id'].isin(usuarios_cluster)]['item_id'].values
    
    # Contar géneros
    genero_counts = {}
    for idx, genero_col in enumerate(genero_cols):
        pelis_genero = peliculas[peliculas['item_id'].isin(pelis_cluster) & (peliculas[genero_col] == 1)]
        genero_counts[generos_nombres.get(idx, f'Género {idx}')] = len(pelis_genero)
    
    # Ordenar y tomar top 10
    genero_counts = dict(sorted(genero_counts.items(), key=lambda x: x[1], reverse=True)[:10])
    
    # Plotear
    ax = axes[cluster_id]
    barras = ax.barh(list(genero_counts.keys()), list(genero_counts.values()), color=sns.color_palette("husl", 3)[cluster_id])
    ax.set_xlabel('Número de Películas Vistas', fontsize=11, fontweight='bold')
    ax.set_title(f'Cluster {cluster_id}: {nombres_clusters[cluster_id]}', fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    
    # Añadir valores en barras
    for i, v in enumerate(genero_counts.values()):
        ax.text(v + 1, i, str(v), va='center', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(ruta_analisis, 'generos_por_cluster.png'), dpi=300, bbox_inches='tight')
print(f"✓ Guardado: generos_por_cluster.png\n")
plt.close()

# GRÁFICA 2: PELÍCULAS MÁS VISTAS POR CLUSTER
print("Generando gráfica 2: Películas más vistas por cluster...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Top 10 Películas Más Vistas por Cluster', fontsize=16, fontweight='bold', y=1.00)

for cluster_id in range(3):
    # Obtener usuarios del cluster
    usuarios_cluster = usuarios_features[usuarios_features['cluster'] == cluster_id]['user_id'].values
    
    # Películas vistas y contar
    pelis_vistas = ratings[ratings['user_id'].isin(usuarios_cluster)]['item_id'].value_counts().head(10)
    
    # Obtener títulos
    titulos = []
    for peli_id in pelis_vistas.index:
        titulo = peliculas[peliculas['item_id'] == peli_id]['titulo'].values
        if len(titulo) > 0:
            titulos.append(titulo[0][:35])  # Limitar a 35 caracteres
    
    # Plotear
    ax = axes[cluster_id]
    barras = ax.bar(range(len(titulos)), pelis_vistas.values, color=sns.color_palette("husl", 3)[cluster_id])
    ax.set_xticks(range(len(titulos)))
    ax.set_xticklabels(titulos, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Número de Visualizaciones', fontsize=11, fontweight='bold')
    ax.set_title(f'Cluster {cluster_id}: {nombres_clusters[cluster_id]}', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Añadir valores en barras
    for i, v in enumerate(pelis_vistas.values):
        ax.text(i, v + 0.5, str(v), ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(ruta_analisis, 'peliculas_populares_por_cluster.png'), dpi=300, bbox_inches='tight')
print(f"✓ Guardado: peliculas_populares_por_cluster.png\n")
plt.close()

# GRÁFICA 3: DISTRIBUCIÓN DE RATINGS POR CLUSTER
print("Generando gráfica 3: Distribución de ratings por cluster...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Distribución de Ratings por Cluster', fontsize=16, fontweight='bold', y=1.02)

for cluster_id in range(3):
    usuarios_cluster = usuarios_features[usuarios_features['cluster'] == cluster_id]['user_id'].values
    ratings_cluster = ratings[ratings['user_id'].isin(usuarios_cluster)]['rating'].values
    
    ax = axes[cluster_id]
    ax.hist(ratings_cluster, bins=5, edgecolor='black', color=sns.color_palette("husl", 3)[cluster_id], alpha=0.7)
    ax.set_xlabel('Rating', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frecuencia', fontsize=11, fontweight='bold')
    ax.set_title(f'Cluster {cluster_id}: {nombres_clusters[cluster_id]}', fontsize=12, fontweight='bold')
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.grid(axis='y', alpha=0.3)
    
    # Estadísticas
    media = np.mean(ratings_cluster)
    ax.axvline(media, color='red', linestyle='--', linewidth=2, label=f'Media: {media:.2f}')
    ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(ruta_analisis, 'distribucion_ratings_por_cluster.png'), dpi=300, bbox_inches='tight')
print(f"✓ Guardado: distribucion_ratings_por_cluster.png\n")
plt.close()

# TABLA RESUMEN
print("Generando tabla de resumen...\n")

resumen_data = []
for cluster_id in range(3):
    usuarios_cluster = usuarios_features[usuarios_features['cluster'] == cluster_id]
    num_usuarios = len(usuarios_cluster)
    edad_media = usuarios_cluster['edad'].mean()
    rating_medio = usuarios_cluster['rating_promedio'].mean()
    movies_media = usuarios_cluster['movies_vistos'].mean()
    
    resumen_data.append({
        'Cluster': cluster_id,
        'Nombre': nombres_clusters[cluster_id],
        'Usuarios': num_usuarios,
        'Edad Media': f"{edad_media:.1f}",
        'Rating Medio': f"{rating_medio:.2f}",
        'Películas Media': f"{movies_media:.0f}"
    })

resumen_df = pd.DataFrame(resumen_data)

fig, ax = plt.subplots(figsize=(12, 3))
ax.axis('tight')
ax.axis('off')

tabla = ax.table(cellText=resumen_df.values, colLabels=resumen_df.columns, 
                cellLoc='center', loc='center', colColours=['#e8e8e8']*len(resumen_df.columns))
tabla.auto_set_font_size(False)
tabla.set_fontsize(11)
tabla.scale(1, 2.5)

plt.title('Resumen de Características por Cluster', fontsize=14, fontweight='bold', pad=20)
plt.savefig(os.path.join(ruta_analisis, 'resumen_clusters.png'), dpi=300, bbox_inches='tight')
print(f"✓ Guardado: resumen_clusters.png\n")
plt.close()

print("="*70)
print("✅ TODAS LAS GRÁFICAS GENERADAS CORRECTAMENTE")
print("="*70)
print("\nArchivos generados:")
print("  1. generos_por_cluster.png")
print("  2. peliculas_populares_por_cluster.png")
print("  3. distribucion_ratings_por_cluster.png")
print("  4. resumen_clusters.png")
print(f"\nUbicación: {ruta_analisis}\n")

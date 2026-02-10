import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

#? Análisis de clustering de usuarios basado en preferencias de películas
#? Aplica K-Means, DBSCAN y visualiza los clusters

ruta_actual = os.path.dirname(os.path.abspath(__file__))
ruta_raiz = os.path.dirname(ruta_actual)
ruta_data = os.path.join(ruta_raiz, 'DATA')
ruta_analisis = os.path.join(ruta_raiz, 'analisis')

print("Extrayendo características de usuarios...\n")

#! Cargar datos limpios
usuarios = pd.read_csv(os.path.join(ruta_data, 'usuarios_limpio.csv'))
peliculas = pd.read_csv(os.path.join(ruta_data, 'peliculas_limpio.csv'))
ratings = pd.read_csv(os.path.join(ruta_data, 'ratings_limpio.csv'))

#! Obtener lista de géneros
genero_cols = [col for col in peliculas.columns if col.startswith('genero_')]
print(f"✓ {len(genero_cols)} géneros encontrados\n")

#! Crear matriz usuario-película con ratings
matriz_usuario_pelicula = ratings.pivot_table(
    index='user_id',
    columns='item_id',
    values='rating',
    fill_value=0
)

print(f"✓ Matriz usuario-película creada: {matriz_usuario_pelicula.shape}\n")

#! Extraer características por usuario

# 1. Rating promedio general por usuario
usuarios_features = usuarios[['user_id', 'edad', 'genero', 'ocupacion']].copy()
usuarios_features['rating_promedio'] = ratings.groupby('user_id')['rating'].mean().values

# 2. Número de películas vistas
usuarios_features['movies_vistos'] = ratings.groupby('user_id').size().values

# 3. Rating promedio por género de película (vectorizado)
# Merge ratings con géneros de películas
ratings_con_generos = ratings.merge(peliculas[['item_id'] + genero_cols], on='item_id')

# Para cada género, calcular rating promedio por usuario
for idx, genero_col in enumerate(genero_cols):
    nombre_genero = f"rating_genero_{idx}"
    
    # Filtrar solo ratings de películas con este género
    ratings_genero = ratings_con_generos[ratings_con_generos[genero_col] == 1]
    
    # Rating promedio por usuario en este género
    rating_por_gen = ratings_genero.groupby('user_id')['rating'].mean().reindex(usuarios_features['user_id'], fill_value=0)
    
    usuarios_features[nombre_genero] = rating_por_gen.values

print(f"✓ Características extraídas: {usuarios_features.shape[1]} features por usuario\n")

#! Mostrar características
print("=== PRIMEROS 5 USUARIOS ===")
print(usuarios_features.head())
print("\n=== ESTADÍSTICAS DE FEATURES ===")
print(usuarios_features.describe())
print("\n")

#! Guardar features
usuarios_features.to_csv(os.path.join(ruta_data, 'usuarios_features.csv'), index=False)
print("✓ Características guardadas en usuarios_features.csv")
print("\n")

#! Preparar datos para clustering
print("=== CLUSTERING DE USUARIOS ===\n")

#* Seleccionar solo features numéricas para clustering
features_numericas = usuarios_features.select_dtypes(include=[np.number]).drop('user_id', axis=1)

#* Normalizar features
scaler = StandardScaler()
features_normalizadas = scaler.fit_transform(features_numericas)

print(f"✓ Datos normalizados: {features_normalizadas.shape}\n")

#! K-MEANS CLUSTERING
print("--- K-MEANS ---")

#* Encontrar número óptimo de clusters con el método del codo
inercias = []
silhuetas = []
rangos_k = range(2, 11)

for k in rangos_k:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(features_normalizadas)
    inercias.append(kmeans.inertia_)
    silhuetas.append(silhouette_score(features_normalizadas, kmeans.labels_))

#* Usar k=3 como punto de balance (común en recomendaciones)
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
clusters = kmeans.fit_predict(features_normalizadas)
silhueta = silhouette_score(features_normalizadas, clusters)

usuarios_features['cluster'] = clusters
print(f"✓ K-Means con k={n_clusters}")
print(f"✓ Silhouette Score: {silhueta:.4f}")
print(f"✓ Distribución de clusters: {np.bincount(clusters)}\n")

#! CARACTERIZACIÓN DE CLUSTERS
print("--- CARACTERIZACIÓN DE CLUSTERS ---\n")

for cluster_id in range(n_clusters):
    usuarios_cluster = usuarios_features[usuarios_features['cluster'] == cluster_id]
    print(f"Cluster {cluster_id}: {len(usuarios_cluster)} usuarios")
    print(f"  Edad promedio: {usuarios_cluster['edad'].mean():.1f} años")
    print(f"  Rating promedio: {usuarios_cluster['rating_promedio'].mean():.2f}/5.0")
    print(f"  Películas vistas promedio: {usuarios_cluster['movies_vistos'].mean():.0f}")
    
    #* Top 3 géneros favoritos
    generos_cols = [col for col in usuarios_cluster.columns if col.startswith('rating_genero')]
    promedio_generos = usuarios_cluster[generos_cols].mean()
    top_generos = promedio_generos.nlargest(3)
    print(f"  Top 3 géneros: {', '.join([f'{genre_num} ({rating:.2f})' for genre_num, rating in top_generos.items()])}")
    print()

#! Guardar clustering
usuarios_features.to_csv(os.path.join(ruta_data, 'usuarios_clusters.csv'), index=False)
print("✓ Clustering guardado en usuarios_clusters.csv\n")

#! VISUALIZACIÓN DE CLUSTERS
print("--- GENERANDO GRÁFICAS ---\n")

#* Reducir dimensionalidad con PCA para visualizar
pca = PCA(n_components=2)
features_pca = pca.fit_transform(features_normalizadas)

print(f"✓ PCA: {pca.explained_variance_ratio_.sum():.2%} varianza explicada\n")

#* Gráfica K-Means
fig, ax = plt.subplots(figsize=(10, 7))
scatter = ax.scatter(features_pca[:, 0], features_pca[:, 1], 
                     c=clusters, cmap='viridis', s=50, alpha=0.6, edgecolors='k', linewidth=0.5)
ax.set_xlabel(f'PCA 1 ({pca.explained_variance_ratio_[0]:.1%})')
ax.set_ylabel(f'PCA 2 ({pca.explained_variance_ratio_[1]:.1%})')
ax.set_title('Clustering K-Means (K=3)')
plt.colorbar(scatter, ax=ax, label='Cluster')
plt.tight_layout()
plt.savefig(os.path.join(ruta_analisis, 'kmeans_clusters.png'), dpi=300, bbox_inches='tight')
plt.close()

print("✓ Gráfica guardada: kmeans_clusters.png\n")

#! GUARDAR MODELO PARA RECOMENDACIONES
print("=== GUARDANDO MODELO PARA RECOMENDACIONES ===\n")

joblib.dump(kmeans, os.path.join(ruta_data, 'modelo_kmeans.pkl'))
joblib.dump(scaler, os.path.join(ruta_data, 'scaler.pkl'))
print("✓ Modelo y scaler guardados\n")

#! FUNCIONES DE RECOMENDACIÓN

def obtener_recomendaciones(user_id, ratings, peliculas, usuarios_features, n_recomendaciones=5):
    """
    Obtiene recomendaciones de 5 películas para un usuario basadas en su cluster.
    
    Parámetros:
    - user_id: ID del usuario
    - ratings: DataFrame de ratings
    - peliculas: DataFrame de películas
    - usuarios_features: DataFrame con clusters asignados
    - n_recomendaciones: número de películas a recomendar (default 5)
    
    Retorna:
    - DataFrame con las películas recomendadas
    """
    
    #* Obtener cluster del usuario
    usuario_cluster = usuarios_features[usuarios_features['user_id'] == user_id]['cluster'].values[0]
    
    #* Obtener usuarios del mismo cluster
    usuarios_mismo_cluster = usuarios_features[usuarios_features['cluster'] == usuario_cluster]['user_id'].values
    
    #* Películas que ha visto el usuario
    pelis_vistas = ratings[ratings['user_id'] == user_id]['item_id'].values
    
    #* Ratings de usuarios del cluster en películas que nuestro usuario no vio
    ratings_cluster_no_visto = ratings[
        (ratings['user_id'].isin(usuarios_mismo_cluster)) & 
        (~ratings['item_id'].isin(pelis_vistas))
    ]
    
    #* Calcular rating promedio por película en el cluster
    rating_promedio_pelis = ratings_cluster_no_visto.groupby('item_id')['rating'].mean().sort_values(ascending=False)
    
    #* Obtener top N películas
    pelis_recomendadas = rating_promedio_pelis.head(n_recomendaciones).index.tolist()
    
    #* Obtener detalles de películas
    recomendaciones = peliculas[peliculas['item_id'].isin(pelis_recomendadas)][['item_id', 'titulo']].copy()
    recomendaciones['rating_cluster'] = recomendaciones['item_id'].map(rating_promedio_pelis)
    
    return recomendaciones, usuario_cluster

#! GENERAR RECOMENDACIONES PARA EJEMPLO DE USUARIOS
print("=== RECOMENDACIONES PERSONALIZADAS === \n")

#* Seleccionar 3 usuarios de ejemplo, uno de cada cluster
usuarios_ejemplo = []
for cluster_id in range(3):
    usuario = usuarios_features[usuarios_features['cluster'] == cluster_id].iloc[0]['user_id']
    usuarios_ejemplo.append(usuario)

#* Generar recomendaciones para cada usuario de ejemplo
for user_id in usuarios_ejemplo:
    recomendaciones, cluster = obtener_recomendaciones(user_id, ratings, peliculas, usuarios_features, n_recomendaciones=5)
    
    print(f"Usuario {user_id} (Cluster {cluster}):")
    print("  Películas recomendadas:")
    for idx, (_, row) in enumerate(recomendaciones.iterrows(), 1):
        print(f"    {idx}. {row['titulo']} (Rating cluster: {row['rating_cluster']:.2f})")
    print()

print("✓ Recomendaciones generadas\n")

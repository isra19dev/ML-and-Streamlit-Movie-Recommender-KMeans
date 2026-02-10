import pandas as pd
import numpy as np
import os
import joblib

#? Script de prueba para validar el sistema de recomendaciones

ruta_actual = os.path.dirname(os.path.abspath(__file__))
ruta_raiz = os.path.dirname(ruta_actual)
ruta_data = os.path.join(ruta_raiz, 'DATA')

print("=== TEST DEL SISTEMA DE RECOMENDACIONES ===\n")

#! Cargar datos
print("Cargando datos...\n")
ratings = pd.read_csv(os.path.join(ruta_data, 'ratings_limpio.csv'))
peliculas = pd.read_csv(os.path.join(ruta_data, 'peliculas_limpio.csv'))
usuarios_features = pd.read_csv(os.path.join(ruta_data, 'usuarios_clusters.csv'))

#! Cargar modelo
kmeans = joblib.load(os.path.join(ruta_data, 'modelo_kmeans.pkl'))
scaler = joblib.load(os.path.join(ruta_data, 'scaler.pkl'))

print("✓ Datos cargados\n")

#! Función para asignar nuevo usuario a cluster
def asignar_cluster_nuevo_usuario(edad, usuarios_vistos, ratings_dados, usuarios_features, ratings, peliculas, scaler, kmeans):
    """
    Asigna un nuevo usuario a un cluster basándose en sus ratings.
    
    Parámetros:
    - edad: edad del usuario
    - usuarios_vistos: lista de item_ids que vio
    - ratings_dados: lista de ratings correspondientes (mismo índice)
    - usuarios_features: DataFrame con features de usuarios ya clustered
    - ratings: DataFrame de ratings
    - peliculas: DataFrame de películas
    - scaler: StandardScaler entrenado
    - kmeans: Modelo KMeans entrenado
    """
    
    genero_cols = [col for col in peliculas.columns if col.startswith('genero_')]
    
    #* Crear features para el nuevo usuario
    features_nuevo = {}
    features_nuevo['edad'] = edad
    
    # Rating promedio
    features_nuevo['rating_promedio'] = np.mean(ratings_dados)
    
    # Películas vistas
    features_nuevo['movies_vistos'] = len(usuarios_vistos)
    
    # Rating promedio por género
    for idx, genero_col in enumerate(genero_cols):
        nombre_genero = f"rating_genero_{idx}"
        
        # Películas que vio del género actual
        pelis_genero = peliculas[peliculas['item_id'].isin(usuarios_vistos) & (peliculas[genero_col] == 1)]['item_id'].values
        
        if len(pelis_genero) > 0:
            # Ratings en ese género
            ratings_gen = [ratings_dados[i] for i in range(len(usuarios_vistos)) if usuarios_vistos[i] in pelis_genero]
            features_nuevo[nombre_genero] = np.mean(ratings_gen) if ratings_gen else 0
        else:
            features_nuevo[nombre_genero] = 0
    
    #* Normalizar features
    features_df = pd.DataFrame([features_nuevo])
    features_escaladas = scaler.transform(features_df)
    
    #* Predecir cluster
    cluster_predicho = kmeans.predict(features_escaladas)[0]
    
    return cluster_predicho, features_nuevo

#! Función para obtener recomendaciones
def obtener_recomendaciones(user_id, ratings, peliculas, usuarios_features, cluster, n_recomendaciones=5):
    """Obtiene recomendaciones para un usuario dado su cluster."""
    
    #* Obtener usuarios del mismo cluster
    usuarios_mismo_cluster = usuarios_features[usuarios_features['cluster'] == cluster]['user_id'].values
    
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
    
    return recomendaciones

#! TEST 1: Usuario nuevo con gustos de acción
print("--- TEST 1: Usuario nuevo (Amante de películas de acción) ---\n")

# Simulamos que vio varias películas de acción y les dio buenas calificaciones
movies_vistas_test1 = [50, 181, 258, 294, 286]  # Películas que asumimos son de acción
ratings_test1 = [5, 4, 5, 4, 5]
edad_test1 = 28

cluster_test1, features_test1 = asignar_cluster_nuevo_usuario(
    edad_test1, movies_vistas_test1, ratings_test1, usuarios_features, ratings, peliculas, scaler, kmeans
)

print(f"Nuevo usuario asignado al Cluster {cluster_test1}")
print(f"Rating promedio: {features_test1['rating_promedio']:.2f}/5.0")
print(f"Películas vistas: {features_test1['movies_vistos']}\n")

# Usar user_id ficticio para obtener recomendaciones (solo para test)
# Usamos un user_id que sabemos que está en el cluster
usuario_cluster = usuarios_features[usuarios_features['cluster'] == cluster_test1].iloc[0]['user_id']
print(f"Generando recomendaciones basadas en usuarios similares (Cluster {cluster_test1}):\n")

recomendaciones_test1 = obtener_recomendaciones(usuario_cluster, ratings, peliculas, usuarios_features, cluster_test1)

print("Películas recomendadas:")
for idx, (_, row) in enumerate(recomendaciones_test1.iterrows(), 1):
    print(f"  {idx}. {row['titulo']} (Rating en cluster: {row['rating_cluster']:.2f})")

print("\n")

#! TEST 2: Usuario hardcore cinéfilo
print("--- TEST 2: Usuario nuevo (Cinéfilo - muchas películas vistas) ---\n")

# Usuario que ha visto muchas películas
movies_vistas_test2 = list(range(100, 150))
ratings_test2 = [3, 4, 3, 2, 4] * 10  # Mix de ratings
edad_test2 = 45

cluster_test2, features_test2 = asignar_cluster_nuevo_usuario(
    edad_test2, movies_vistas_test2, ratings_test2, usuarios_features, ratings, peliculas, scaler, kmeans
)

print(f"Nuevo usuario asignado al Cluster {cluster_test2}")
print(f"Rating promedio: {features_test2['rating_promedio']:.2f}/5.0")
print(f"Películas vistas: {features_test2['movies_vistos']}\n")

usuario_cluster2 = usuarios_features[usuarios_features['cluster'] == cluster_test2].iloc[0]['user_id']
print(f"Generando recomendaciones para Cluster {cluster_test2}:\n")

recomendaciones_test2 = obtener_recomendaciones(usuario_cluster2, ratings, peliculas, usuarios_features, cluster_test2)

print("Películas recomendadas:")
for idx, (_, row) in enumerate(recomendaciones_test2.iterrows(), 1):
    print(f"  {idx}. {row['titulo']} (Rating en cluster: {row['rating_cluster']:.2f})")

print("\n")

#! TEST 3: Usuario selectivo
print("--- TEST 3: Usuario nuevo (Selectivo - pocas películas, muy exigente) ---\n")

movies_vistas_test3 = [1, 15, 25]
ratings_test3 = [5, 5, 4]
edad_test3 = 32

cluster_test3, features_test3 = asignar_cluster_nuevo_usuario(
    edad_test3, movies_vistas_test3, ratings_test3, usuarios_features, ratings, peliculas, scaler, kmeans
)

print(f"Nuevo usuario asignado al Cluster {cluster_test3}")
print(f"Rating promedio: {features_test3['rating_promedio']:.2f}/5.0")
print(f"Películas vistas: {features_test3['movies_vistos']}\n")

usuario_cluster3 = usuarios_features[usuarios_features['cluster'] == cluster_test3].iloc[0]['user_id']
print(f"Generando recomendaciones para Cluster {cluster_test3}:\n")

recomendaciones_test3 = obtener_recomendaciones(usuario_cluster3, ratings, peliculas, usuarios_features, cluster_test3)

print("Películas recomendadas:")
for idx, (_, row) in enumerate(recomendaciones_test3.iterrows(), 1):
    print(f"  {idx}. {row['titulo']} (Rating en cluster: {row['rating_cluster']:.2f})")

print("\n✓ Todos los tests completados correctamente!")

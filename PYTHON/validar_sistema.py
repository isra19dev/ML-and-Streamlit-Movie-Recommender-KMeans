import pandas as pd
import numpy as np
import os
import joblib

#? Script para correr los ejemplos de prueba y validar el sistema

ruta_actual = os.path.dirname(os.path.abspath(__file__))
ruta_raiz = os.path.dirname(ruta_actual)
ruta_data = os.path.join(ruta_raiz, 'DATA')

print("\n" + "="*70)
print("  VALIDACIÓN DEL SISTEMA DE RECOMENDACIONES NETFLIX")
print("="*70 + "\n")

# Cargar datos
ratings = pd.read_csv(os.path.join(ruta_data, 'ratings_limpio.csv'))
peliculas = pd.read_csv(os.path.join(ruta_data, 'peliculas_limpio.csv'))
usuarios_features = pd.read_csv(os.path.join(ruta_data, 'usuarios_clusters.csv'))
kmeans = joblib.load(os.path.join(ruta_data, 'modelo_kmeans.pkl'))
scaler = joblib.load(os.path.join(ruta_data, 'scaler.pkl'))

def asignar_cluster(edad, pelis_vistas, ratings_dados):
    """Asigna un usuario nuevo a su cluster más cercano."""
    
    genero_cols = [col for col in peliculas.columns if col.startswith('genero_')]
    
    features_nuevo = {'edad': edad}
    features_nuevo['rating_promedio'] = np.mean(ratings_dados) if ratings_dados else 0
    features_nuevo['movies_vistos'] = len(pelis_vistas)
    
    for idx, genero_col in enumerate(genero_cols):
        nombre_genero = f"rating_genero_{idx}"
        pelis_genero = peliculas[peliculas['item_id'].isin(pelis_vistas) & (peliculas[genero_col] == 1)]['item_id'].values
        
        if len(pelis_genero) > 0:
            ratings_gen = [ratings_dados[i] for i in range(len(pelis_vistas)) if pelis_vistas[i] in pelis_genero]
            features_nuevo[nombre_genero] = np.mean(ratings_gen) if ratings_gen else 0
        else:
            features_nuevo[nombre_genero] = 0
    
    features_df = pd.DataFrame([features_nuevo])
    features_escaladas = scaler.transform(features_df)
    cluster_predicho = kmeans.predict(features_escaladas)[0]
    
    return cluster_predicho, features_nuevo

def obtener_recomendaciones_streamlit(pelis_vistas, cluster, ratings_dados, edad, n_recomendaciones=5):
    """
    Obtiene recomendaciones personalizadas basadas en:
    1. Usuario más similar en el cluster
    2. Películas que vieron usuarios similares
    3. Películas no vistas por el usuario actual
    """
    
    usuarios_mismo_cluster = usuarios_features[usuarios_features['cluster'] == cluster].copy()
    
    if len(usuarios_mismo_cluster) == 0:
        return pd.DataFrame()
    
    # 1. Encontrar usuarios más similares basado en edad, rating promedio y movies vistos
    usuarios_mismo_cluster['diferencia_edad'] = (usuarios_mismo_cluster['edad'] - edad).abs()
    usuarios_mismo_cluster['diferencia_rating'] = (usuarios_mismo_cluster['rating_promedio'] - np.mean(ratings_dados)).abs()
    usuarios_mismo_cluster['diferencia_movies'] = (usuarios_mismo_cluster['movies_vistos'] - len(pelis_vistas)).abs()
    
    # Normalizar diferencias
    usuarios_mismo_cluster['edad_norm'] = usuarios_mismo_cluster['diferencia_edad'] / (usuarios_mismo_cluster['diferencia_edad'].max() + 1)
    usuarios_mismo_cluster['rating_norm'] = usuarios_mismo_cluster['diferencia_rating'] / (usuarios_mismo_cluster['diferencia_rating'].max() + 1)
    usuarios_mismo_cluster['movies_norm'] = usuarios_mismo_cluster['diferencia_movies'] / (usuarios_mismo_cluster['diferencia_movies'].max() + 1)
    
    # Calcular similitud ponderada
    usuarios_mismo_cluster['similitud'] = (
        usuarios_mismo_cluster['edad_norm'] * 0.4 +
        usuarios_mismo_cluster['rating_norm'] * 0.35 +
        usuarios_mismo_cluster['movies_norm'] * 0.25
    )
    
    # Obtener top 10 usuarios similares
    usuarios_similares = usuarios_mismo_cluster.nsmallest(10, 'similitud')['user_id'].values
    
    # 2. Obtener películas de usuarios similares no vistas
    ratings_de_similares = ratings[
        (ratings['user_id'].isin(usuarios_similares)) & 
        (~ratings['item_id'].isin(pelis_vistas)) &
        (ratings['rating'] >= 4)
    ]
    
    if len(ratings_de_similares) == 0:
        ratings_de_similares = ratings[
            (ratings['user_id'].isin(usuarios_similares)) & 
            (~ratings['item_id'].isin(pelis_vistas))
        ]
    
    # 3. Calcular estadísticas
    recomendaciones_df = ratings_de_similares.groupby('item_id').agg({
        'rating': ['mean', 'count']
    }).reset_index()
    
    recomendaciones_df.columns = ['item_id', 'rating_promedio', 'num_usuarios']
    recomendaciones_df = recomendaciones_df[recomendaciones_df['num_usuarios'] >= 2]
    recomendaciones_df = recomendaciones_df.sort_values(['rating_promedio', 'num_usuarios'], ascending=False)
    
    pelis_recomendadas = recomendaciones_df.head(n_recomendaciones)['item_id'].tolist()
    
    # 4. Obtener info de películas
    recomendaciones = peliculas[peliculas['item_id'].isin(pelis_recomendadas)][['item_id', 'titulo']].copy()
    recomendaciones['rating'] = recomendaciones['item_id'].map(
        dict(zip(recomendaciones_df['item_id'], recomendaciones_df['rating_promedio']))
    )
    
    return recomendaciones.sort_values('rating', ascending=False)

# Información de clusters
nombres_clusters = {0: "Selectivos", 1: "Críticos Activos", 2: "Cinéfilos"}

# EJEMPLO 1: Usuario Selectivo
print("┌" + "─"*68 + "┐")
print("│ EJEMPLO 1: Usuario Selectivo (Pocas películas, altas puntuaciones)     │")
print("└" + "─"*68 + "┘\n")

print("PARÁMETROS DE ENTRADA:")
print("  Edad: 32 años")
print("  Películas vistas:")

# Buscar IDs de las películas por título
peliculas_ejemplo1 = ['Good Will Hunting', 'The Shawshank Redemption', 'Pulp Fiction']
ids_ejemplo1 = []
for titulo in peliculas_ejemplo1:
    peli = peliculas[peliculas['titulo'].str.contains(titulo, case=False, na=False)]
    if len(peli) > 0:
        ids_ejemplo1.append(peli.iloc[0]['item_id'])

ratings_ejemplo1 = [5, 5, 4]

print(f"    - Título que contiene '{peliculas_ejemplo1[0]}' → ⭐ {ratings_ejemplo1[0]}")
print(f"    - Título que contiene '{peliculas_ejemplo1[1]}' → ⭐ {ratings_ejemplo1[1]}")
print(f"    - Título que contiene '{peliculas_ejemplo1[2]}' → ⭐ {ratings_ejemplo1[2]}\n")

cluster1, features1 = asignar_cluster(32, ids_ejemplo1, ratings_ejemplo1)
print("SALIDA ESPERADA:")
print(f"  ✅ Cluster Asignado: {cluster1} - '{nombres_clusters.get(cluster1, 'Desconocido')}'")
print(f"     Rating promedio: {features1['rating_promedio']:.2f}/5")
print(f"     Películas vistas: {features1['movies_vistos']}\n")

recom1 = obtener_recomendaciones_streamlit(ids_ejemplo1, cluster1, ratings_ejemplo1, 32)
print("  🎬 Top 5 Recomendaciones:")
for idx, (_, row) in enumerate(recom1.head(5).iterrows(), 1):
    stars = "⭐" * int(round(row['rating']))
    print(f"     {idx}. {row['titulo'][:40]:40} {stars} {row['rating']:.2f}")

print("\n" + "="*70 + "\n")

# EJEMPLO 2: Usuario Crítico Activo
print("┌" + "─"*68 + "┐")
print("│ EJEMPLO 2: Usuario Crítico Activo (Muchas películas, rating medio)    │")
print("└" + "─"*68 + "┘\n")

print("PARÁMETROS DE ENTRADA:")
print("  Edad: 34 años")
print("  Películas vistas (8+ películas con ratings variados):\n")

# Generar múltiples películas de acción/sci-fi
peliculas_keywords = ['Terminator', 'Total Recall', 'Predator', 'RoboCop', 'Running', 'Volume', 'Steel', 'Navy']
ids_ejemplo2 = []
ratings_ejemplo2 = [4, 3, 5, 3, 2, 3, 3, 3]

for i, keyword in enumerate(peliculas_keywords[:8]):
    pelis_encontradas = peliculas[peliculas['titulo'].str.contains(keyword, case=False, na=False)]
    if len(pelis_encontradas) > 0:
        ids_ejemplo2.append(pelis_encontradas.iloc[0]['item_id'])
        print(f"    - {pelis_encontradas.iloc[0]['titulo'][:40]:40} → ⭐ {ratings_ejemplo2[i]}")

print()
cluster2, features2 = asignar_cluster(34, ids_ejemplo2, ratings_ejemplo2)
print("SALIDA ESPERADA:")
print(f"  ✅ Cluster Asignado: {cluster2} - '{nombres_clusters.get(cluster2, 'Desconocido')}'")
print(f"     Rating promedio: {features2['rating_promedio']:.2f}/5")
print(f"     Películas vistas: {features2['movies_vistos']}\n")

recom2 = obtener_recomendaciones_streamlit(ids_ejemplo2, cluster2, ratings_ejemplo2, 34)
print("  🎬 Top 5 Recomendaciones:")
for idx, (_, row) in enumerate(recom2.head(5).iterrows(), 1):
    stars = "⭐" * int(round(row['rating']))
    print(f"     {idx}. {row['titulo'][:40]:40} {stars} {row['rating']:.2f}")

print("\n" + "="*70 + "\n")

# EJEMPLO 3: Usuario Cinéfilo
print("┌" + "─"*68 + "┐")
print("│ EJEMPLO 3: Usuario Cinéfilo (Muchas películas, buenas puntuaciones)   │")
print("└" + "─"*68 + "┘\n")

print("PARÁMETROS DE ENTRADA:")
print("  Edad: 40 años")
print("  Películas vistas (15+ películas con buenas puntuaciones):\n")

peliculas_keywords3 = ['Blade Runner', 'Space Odyssey', 'Alien', 'Terminator', 'Tron', 'Fifth Element', 'Dark City', 'Brazil', 'Akira', 'Ghost', 'Total Recall', 'RoboCop', 'Escape from', 'Road Warrior', 'Star Wars']
ids_ejemplo3 = []
ratings_ejemplo3 = [5, 5, 5, 4, 4, 4, 4, 5, 5, 5, 4, 4, 4, 5, 4]

for i, keyword in enumerate(peliculas_keywords3):
    pelis_encontradas = peliculas[peliculas['titulo'].str.contains(keyword, case=False, na=False)]
    if len(pelis_encontradas) > 0:
        ids_ejemplo3.append(pelis_encontradas.iloc[0]['item_id'])
        print(f"    - {pelis_encontradas.iloc[0]['titulo'][:40]:40} → ⭐ {ratings_ejemplo3[i]}")

print()
cluster3, features3 = asignar_cluster(40, ids_ejemplo3, ratings_ejemplo3)
print("SALIDA ESPERADA:")
print(f"  ✅ Cluster Asignado: {cluster3} - '{nombres_clusters.get(cluster3, 'Desconocido')}'")
print(f"     Rating promedio: {features3['rating_promedio']:.2f}/5")
print(f"     Películas vistas: {features3['movies_vistos']}\n")

recom3 = obtener_recomendaciones_streamlit(ids_ejemplo3, cluster3, ratings_ejemplo3, 40)
print("  🎬 Top 5 Recomendaciones:")
for idx, (_, row) in enumerate(recom3.head(5).iterrows(), 1):
    stars = "⭐" * int(round(row['rating']))
    print(f"     {idx}. {row['titulo'][:40]:40} {stars} {row['rating']:.2f}")

print("\n" + "="*70)
print("\n✅ PRUEBA COMPLETADA")
print("\nNota: Los IDs de película se buscan automáticamente por palabra clave.")
print("Por lo tanto, los números exactos de clusters pueden variar ligeramente\n")

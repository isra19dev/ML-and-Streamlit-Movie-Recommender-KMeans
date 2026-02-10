import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

#? Script para generar gráfica del método del codo (Elbow Method)

ruta_actual = os.path.dirname(os.path.abspath(__file__))
ruta_raiz = os.path.dirname(ruta_actual)
ruta_data = os.path.join(ruta_raiz, 'DATA')
ruta_analisis = os.path.join(ruta_raiz, 'analisis')

# Crear carpeta si no existe
if not os.path.exists(ruta_analisis):
    os.makedirs(ruta_analisis)

print("="*70)
print("  GENERANDO GRÁFICA DEL MÉTODO DEL CODO (ELBOW METHOD)")
print("="*70 + "\n")

# Cargar datos
print("Cargando datos...\n")
ratings = pd.read_csv(os.path.join(ruta_data, 'ratings_limpio.csv'))
peliculas = pd.read_csv(os.path.join(ruta_data, 'peliculas_limpio.csv'))
usuarios = pd.read_csv(os.path.join(ruta_data, 'usuarios_limpio.csv'))

# Extraer features (igual que en main.py)
print("Extrayendo características de usuarios...")

genero_cols = [col for col in peliculas.columns if col.startswith('genero_')]
usuarios_features = usuarios[['user_id', 'edad']].copy()

#* Calcular rating promedio y películas vistas
ratings_por_usuario = ratings.groupby('user_id').agg({
    'rating': 'mean',
    'item_id': 'count'
}).rename(columns={'rating': 'rating_promedio', 'item_id': 'movies_vistos'})

usuarios_features = usuarios_features.merge(ratings_por_usuario, on='user_id', how='left')

#* Rating promedio por género
for idx, genero_col in enumerate(genero_cols):
    nombre_genero = f"rating_genero_{idx}"
    
    # Películas de este género
    pelis_genero = peliculas[peliculas[genero_col] == 1][['item_id']].copy()
    
    # Ratings de esas películas
    ratings_genero = ratings[ratings['item_id'].isin(pelis_genero['item_id'].values)].copy()
    
    # Agrupar por usuario
    rating_por_genero = ratings_genero.groupby('user_id')['rating'].mean()
    
    usuarios_features[nombre_genero] = usuarios_features['user_id'].map(rating_por_genero)
    usuarios_features[nombre_genero] = usuarios_features[nombre_genero].fillna(0)

print("✓ Características extraídas\n")

# Preparar datos para clustering
X = usuarios_features.iloc[:, 1:].values  # Excluir user_id
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Calcular inercia para diferentes valores de k
print("Entrenando K-Means para k = 1 a 10...")
inercias = []
silhuetas = []
k_values = range(1, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inercias.append(kmeans.inertia_)
    print(f"  k={k}: Inercia = {kmeans.inertia_:.2f}")

print("\nGenerando gráfica del método del codo...")

# Crear gráfica
fig, ax = plt.subplots(figsize=(12, 6))

# Plot principal
ax.plot(k_values, inercias, 'bo-', linewidth=2, markersize=8, label='Inercia')

# Indicar el k óptimo (donde está el error)
# Típicamente es donde la curva cambia de pendiente (elbow)
ax.axvline(x=3, color='red', linestyle='--', linewidth=2, alpha=0.7, label='K=3 (seleccionado)')

# Relleno
ax.fill_between(k_values, inercias, alpha=0.3)

# Etiquetas y títulos
ax.set_xlabel('Número de Clusters (k)', fontsize=12, fontweight='bold')
ax.set_ylabel('Inercia (within-cluster sum of squares)', fontsize=12, fontweight='bold')
ax.set_title('Método del Codo - Análisis de K-Means', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(k_values)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11, loc='upper right')

# Añadir anotaciones
for i, (k, inercia) in enumerate(zip(k_values, inercias)):
    ax.annotate(f'{inercia:.0f}', xy=(k, inercia), xytext=(0, 8), 
                textcoords='offset points', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(ruta_analisis, 'elbow_method.png'), dpi=300, bbox_inches='tight')
print(f"✓ Guardado: elbow_method.png")

# Crear segunda gráfica con tasa de cambio (para identificar mejor el codo)
fig, ax = plt.subplots(figsize=(12, 6))

# Calcular diferencias
diferencias = [inercias[i] - inercias[i+1] for i in range(len(inercias)-1)]
diferencias_segunda = [diferencias[i] - diferencias[i+1] for i in range(len(diferencias)-1)]

# Plot
x_diff = list(k_values)[:-1]
x_diff2 = list(k_values)[:-2]

bars = ax.bar(x_diff, diferencias, alpha=0.7, label='Cambio en inercia', color='steelblue', width=0.4)

ax.axvline(x=3, color='red', linestyle='--', linewidth=2, alpha=0.7, label='K=3 (seleccionado)')

ax.set_xlabel('Número de Clusters (k)', fontsize=12, fontweight='bold')
ax.set_ylabel('Reducción de Inercia', fontsize=12, fontweight='bold')
ax.set_title('Tasa de Cambio de Inercia - Identificación del Codo', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x_diff)
ax.grid(True, alpha=0.3, axis='y')
ax.legend(fontsize=11)

# Anotaciones
for bar, valor in zip(bars, diferencias):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
            f'{valor:.0f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(ruta_analisis, 'elbow_method_tasa_cambio.png'), dpi=300, bbox_inches='tight')
print(f"✓ Guardado: elbow_method_tasa_cambio.png")

plt.close('all')

print("\n" + "="*70)
print("✅ GRÁFICAS DEL MÉTODO DEL CODO GENERADAS CORRECTAMENTE")
print("="*70)
print("\nArchivos creados:")
print("  1. elbow_method.png - Curva de inercia")
print("  2. elbow_method_tasa_cambio.png - Tasa de cambio de inercia")
print(f"\nUbicación: {ruta_analisis}\n")

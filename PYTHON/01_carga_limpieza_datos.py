import pandas as pd
import os

#? Este script carga y limpia los datos del dataset MovieLens
#? Genera 3 archivos CSV limpios que se usarán en análisis posteriores

ruta_actual = os.path.dirname(os.path.abspath(__file__))
ruta_raiz = os.path.dirname(ruta_actual)
ruta_data = os.path.join(ruta_raiz, 'DATA')
ruta_raw = os.path.join(ruta_data, 'raw')

print("Cargando y limpiando datos del dataset MovieLens...\n")

#! Cargar datos originales (desde DATA/raw/)
usuarios = pd.read_csv(os.path.join(ruta_raw, 'u.user'), sep='|', header=None, 
                       names=['user_id', 'edad', 'genero', 'ocupacion', 'codigo_postal'])

peliculas = pd.read_csv(os.path.join(ruta_raw, 'u.item'), sep='|', encoding='latin1', header=None,
                        names=['item_id', 'titulo', 'fecha_estreno', 'fecha_video', 'url_imdb'] + 
                              [f'genero_{i}' for i in range(19)])

ratings = pd.read_csv(os.path.join(ruta_raw, 'u.data'), sep='\t', header=None,
                      names=['user_id', 'item_id', 'rating', 'timestamp'])

print(f"✓ Datos cargados: {len(usuarios)} usuarios, {len(peliculas)} películas, {len(ratings)} ratings\n")

#! Limpieza de datos

#* Usuarios - Convertir tipos de datos
usuarios['edad'] = pd.to_numeric(usuarios['edad'], errors='coerce')
usuarios['genero'] = usuarios['genero'].astype('category')
usuarios['ocupacion'] = usuarios['ocupacion'].astype('category')

#* Películas - Limpiar fechas y eliminar columnas innecesarias
peliculas['fecha_estreno'] = pd.to_datetime(peliculas['fecha_estreno'], errors='coerce')
peliculas = peliculas.drop(['fecha_video', 'url_imdb'], axis=1)

#* Ratings - Convertir timestamp (por si acaso)
ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
ratings['rating'] = pd.to_numeric(ratings['rating'], errors='coerce')

#! Eliminar filas con datos críticos faltantes
usuarios = usuarios.dropna(subset=['user_id', 'edad', 'genero', 'ocupacion'])
ratings = ratings.dropna(subset=['user_id', 'item_id', 'rating'])

print(f"✓ Datos limpios: {len(usuarios)} usuarios, {len(peliculas)} películas, {len(ratings)} ratings\n")

#! Guardar archivos CSV limpios (en DATA/, no en raw/)
usuarios.to_csv(os.path.join(ruta_data, 'usuarios_limpio.csv'), index=False)
peliculas.to_csv(os.path.join(ruta_data, 'peliculas_limpio.csv'), index=False)
ratings.to_csv(os.path.join(ruta_data, 'ratings_limpio.csv'), index=False)

print("✓ Archivos guardados: usuarios_limpio.csv, peliculas_limpio.csv, ratings_limpio.csv")

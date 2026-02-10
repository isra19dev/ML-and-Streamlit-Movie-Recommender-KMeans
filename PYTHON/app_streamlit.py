import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

#? Interfaz Streamlit para sistema de recomendaciones de películas

ruta_actual = os.path.dirname(os.path.abspath(__file__))
ruta_raiz = os.path.dirname(ruta_actual)
ruta_data = os.path.join(ruta_raiz, 'DATA')

#! Configuración de página
st.set_page_config(
    page_title="Sistema de Recomendación Netflix",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Sistema de Recomendación de Películas")
st.markdown("Descubre películas basadas en tus gustos personalizados")

#! Cargar datos y modelo
@st.cache_resource
def cargar_recursos():
    ratings = pd.read_csv(os.path.join(ruta_data, 'ratings_limpio.csv'))
    peliculas = pd.read_csv(os.path.join(ruta_data, 'peliculas_limpio.csv'))
    usuarios_features = pd.read_csv(os.path.join(ruta_data, 'usuarios_clusters.csv'))
    kmeans = joblib.load(os.path.join(ruta_data, 'modelo_kmeans.pkl'))
    scaler = joblib.load(os.path.join(ruta_data, 'scaler.pkl'))
    return ratings, peliculas, usuarios_features, kmeans, scaler

ratings, peliculas, usuarios_features, kmeans, scaler = cargar_recursos()

#! Inicializar session state
if 'peliculas_anyadidas' not in st.session_state:
    st.session_state.peliculas_anyadidas = []

#! Función para asignar cluster
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
    cluster = kmeans.predict(features_escaladas)[0]
    
    return cluster, features_nuevo

#! Función para obtener recomendaciones personalizadas
def obtener_recomendaciones_streamlit(pelis_vistas, cluster, ratings_dados, edad, n_recomendaciones=5):
    """
    Obtiene recomendaciones personalizadas basadas en:
    1. Usuario más similar en el cluster
    2. Películas que vieron usuarios similares
    3. Películas no vistas por el usuario actual
    """
    
    usuarios_mismo_cluster = usuarios_features[usuarios_features['cluster'] == cluster]
    
    if len(usuarios_mismo_cluster) == 0:
        return pd.DataFrame()
    
    # 1. Encontrar usuario más similar basado en edad, rating promedio y movies vistos
    usuarios_mismo_cluster['diferencia_edad'] = (usuarios_mismo_cluster['edad'] - edad).abs()
    usuarios_mismo_cluster['diferencia_rating'] = (usuarios_mismo_cluster['rating_promedio'] - np.mean(ratings_dados)).abs()
    usuarios_mismo_cluster['diferencia_movies'] = (usuarios_mismo_cluster['movies_vistos'] - len(pelis_vistas)).abs()
    
    # Normalizar diferencias (0-1)
    usuarios_mismo_cluster['edad_norm'] = usuarios_mismo_cluster['diferencia_edad'] / (usuarios_mismo_cluster['diferencia_edad'].max() + 1)
    usuarios_mismo_cluster['rating_norm'] = usuarios_mismo_cluster['diferencia_rating'] / (usuarios_mismo_cluster['diferencia_rating'].max() + 1)
    usuarios_mismo_cluster['movies_norm'] = usuarios_mismo_cluster['diferencia_movies'] / (usuarios_mismo_cluster['diferencia_movies'].max() + 1)
    
    # Calcular similitud ponderada (edad 40%, rating 35%, movies 25%)
    usuarios_mismo_cluster['similitud'] = (
        usuarios_mismo_cluster['edad_norm'] * 0.4 +
        usuarios_mismo_cluster['rating_norm'] * 0.35 +
        usuarios_mismo_cluster['movies_norm'] * 0.25
    )
    
    # Obtener top 10 usuarios similares
    usuarios_similares = usuarios_mismo_cluster.nsmallest(10, 'similitud')['user_id'].values
    
    # 2. Obtener películas que vieron estos usuarios similares pero no vio el usuario actual
    ratings_de_similares = ratings[
        (ratings['user_id'].isin(usuarios_similares)) & 
        (~ratings['item_id'].isin(pelis_vistas)) &
        (ratings['rating'] >= 4)  # Solo películas bien valoradas
    ]
    
    if len(ratings_de_similares) == 0:
        # Si no hay películas bien valoradas, usar todas las no vistas
        ratings_de_similares = ratings[
            (ratings['user_id'].isin(usuarios_similares)) & 
            (~ratings['item_id'].isin(pelis_vistas))
        ]
    
    # 3. Calcular rating promedio y contar cuántos la recomiendan
    recomendaciones_df = ratings_de_similares.groupby('item_id').agg({
        'rating': ['mean', 'count']
    }).reset_index()
    
    recomendaciones_df.columns = ['item_id', 'rating_promedio', 'num_usuarios']
    
    # Ordenar por rating pero preferencias múltiples recomendaciones
    recomendaciones_df = recomendaciones_df[recomendaciones_df['num_usuarios'] >= 2]  # Al menos 2 usuarios similares
    recomendaciones_df = recomendaciones_df.sort_values(['rating_promedio', 'num_usuarios'], ascending=False)
    
    pelis_recomendadas = recomendaciones_df.head(n_recomendaciones)['item_id'].tolist()
    
    # 4. Obtener info de películas
    recomendaciones = peliculas[peliculas['item_id'].isin(pelis_recomendadas)][['item_id', 'titulo']].copy()
    recomendaciones['rating'] = recomendaciones['item_id'].map(
        dict(zip(recomendaciones_df['item_id'], recomendaciones_df['rating_promedio']))
    )
    
    return recomendaciones.sort_values('rating', ascending=False)

#! INTERFAZ PRINCIPAL
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Tu Información")
    edad = st.slider("¿Cuántos años tienes?", 13, 80, 30)
    
with col2:
    st.subheader("🎥 Películas")
    st.caption("Busca y selecciona las que has visto")

st.markdown("---")

#! Selección de películas
st.subheader("🔍 Añade tus películas vistas")

#* Inicializar lista de películas en session state si no existe
if 'peliculas_anyadidas' not in st.session_state:
    st.session_state.peliculas_anyadidas = []

#* Búsqueda y autocompletado
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    busqueda = st.text_input("Escribe el nombre de la película", placeholder="ej: Toy, Matrix, Avatar...").strip().lower()
    
    #* Filtrar películas según búsqueda
    if busqueda:
        opciones = peliculas[peliculas['titulo'].str.lower().str.contains(busqueda, na=False, regex=False)]['titulo'].unique().tolist()
    else:
        opciones = peliculas['titulo'].unique().tolist()[:30]  # Primeras 30 si no hay búsqueda
    
    #* Selectbox con autocompletado
    if opciones:
        titulo_pelicula = st.selectbox(
            "Selecciona una película",
            options=opciones,
            label_visibility="collapsed"
        )
    else:
        st.error("❌ No se encontraron películas")
        titulo_pelicula = None

with col2:
    puntuacion = st.slider("Tu puntuación", 1, 5, 3)

with col3:
    st.write("")  # Espacio
    st.write("")  # Espacio
    if st.button("✅ Añadir película", use_container_width=True):
        if titulo_pelicula:
            #* Buscar la película en la base de datos
            pelicula_encontrada = peliculas[
                peliculas['titulo'].str.lower() == titulo_pelicula.lower()
            ]
            
            if len(pelicula_encontrada) > 0:
                movie_id = pelicula_encontrada.iloc[0]['item_id']
                titulo_exacto = pelicula_encontrada.iloc[0]['titulo']
                
                #* Verificar que no esté duplicada
                existe = any(p['titulo'] == titulo_exacto for p in st.session_state.peliculas_anyadidas)
                
                if not existe:
                    st.session_state.peliculas_anyadidas.append({
                        'titulo': titulo_exacto,
                        'id': movie_id,
                        'puntuacion': puntuacion
                    })
                    st.success(f"✓ '{titulo_exacto}' añadida con puntuación {puntuacion}/5")
                else:
                    st.warning(f"⚠️ '{titulo_exacto}' ya está en tu lista")

st.markdown("---")

#* Mostrar películas añadidas
if st.session_state.peliculas_anyadidas:
    st.subheader(f"🎬 Películas añadidas ({len(st.session_state.peliculas_anyadidas)})")
    
    for i, pelicula in enumerate(st.session_state.peliculas_anyadidas):
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            st.write(f"**{pelicula['titulo']}**")
        with col2:
            st.write(f"⭐ {pelicula['puntuacion']}/5")
        with col3:
            if st.button("🗑️", key=f"eliminar_{i}"):
                st.session_state.peliculas_anyadidas.pop(i)
                st.rerun()
else:
    st.info("Aún no has añadido películas. ¡Empieza a añadirlas!")

pelis_seleccionadas = st.session_state.peliculas_anyadidas

st.markdown("---")

#! Generar recomendaciones
if pelis_seleccionadas:
    if st.button("🚀 Generar Recomendaciones", use_container_width=True, type="primary"):
        
        #* Obtener IDs y ratings de películas seleccionadas
        pelis_ids = [p['id'] for p in pelis_seleccionadas]
        ratings_valores = [p['puntuacion'] for p in pelis_seleccionadas]
        
        #* Asignar cluster
        cluster, features = asignar_cluster(edad, pelis_ids, ratings_valores)
        
        #* Obtener recomendaciones personalizadas
        recomendaciones = obtener_recomendaciones_streamlit(pelis_ids, cluster, ratings_valores, edad)
        
        #! Mostrar resultados
        st.markdown("---")
        st.header("✨ Tus Recomendaciones Personalizadas")
        
        #* Información del usuario
        col_info1, col_info2, col_info3 = st.columns(3)
        
        with col_info1:
            nombres_clusters = {0: "Selectivos", 1: "Críticos Activos", 2: "Cinéfilos"}
            st.metric("Tu Grupo", nombres_clusters.get(cluster, f"Cluster {cluster}"))
        
        with col_info2:
            st.metric("Rating Promedio", f"{features['rating_promedio']:.2f} ⭐")
        
        with col_info3:
            st.metric("Películas Vistas", features['movies_vistos'])
        
        st.markdown("---")
        
        #* Mostrar recomendaciones
        if len(recomendaciones) > 0:
            st.subheader(f"📽️ Top {len(recomendaciones)} Películas Recomendadas")
            
            for idx, (_, row) in enumerate(recomendaciones.iterrows(), 1):
                col_num, col_peli, col_rating = st.columns([0.5, 3, 1.2])
                
                with col_num:
                    st.markdown(f"### {idx}")
                
                with col_peli:
                    st.markdown(f"**{row['titulo']}**")
                
                with col_rating:
                    stars = "⭐" * int(round(row['rating']))
                    st.text(f"{stars} {row['rating']:.2f}")
        else:
            st.warning("⚠️ No se encontraron recomendaciones disponibles para tu perfil")
        
        st.markdown("---")
        st.success("✅ ¡Recomendaciones generadas!")

else:
    st.info("👆 Añade al menos una película para obtener recomendaciones personalizadas")

#! Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 12px;'>"
    "Proyecto de Recomendación de Películas - K-Means Clustering | MovieLens Dataset"
    "</div>",
    unsafe_allow_html=True
)

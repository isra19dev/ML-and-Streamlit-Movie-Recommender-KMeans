#  Sistema de Recomendación de Películas - Netflix

Sistema inteligente de recomendación de películas basado en **K-Means clustering** del dataset MovieLens 100K. Agrupa usuarios en 3 perfiles y genera recomendaciones personalizadas.

##  Características

-  **K-Means Clustering**: Segmenta 943 usuarios en 3 grupos
-  **25 Features por usuario**: Edad, rating promedio, películas vistas, preferencias por género
-  **Recomendaciones personalizadas**: Basadas en similitud con usuarios del mismo cluster
-  **Interfaz Streamlit**: App interactiva y amigable
-  **5 Visualizaciones** de análisis por cluster

##  Requisitos

```bash
python >= 3.8
pandas
numpy
scikit-learn
matplotlib
seaborn
streamlit
joblib
```

Instalar todo:
```bash
pip install -r requirements.txt
```

##  Cómo Usar

### 1. Preparar datos (primera vez)
```bash
cd PYTHON
python 01_carga_limpieza_datos.py
```
→ Genera CSVs limpios en `DATA/`

### 2. Entrenar modelo
```bash
python main.py
```
→ Entrena K-Means, guarda modelos, genera gráficas

### 3. Ejecutar app web
```bash
streamlit run app_streamlit.py
```
→ Abre http://localhost:8501

### 4. Validar sistema (test)
```bash
python validar_sistema.py
```
→ Prueba con 3 perfiles de usuario


##  Los 3 Clusters

| Cluster | Nombre | Usuarios | Movies Media | Rating Medio | Perfil |
|---------|--------|----------|--------------|--------------|--------|
| 0 | Selectivos | 312 | 43 | 3.81 | Ven pocas películas pero les dan buenas puntuaciones |
| 1 | Críticos Activos | 258 | 90 | 3.08 | Super activos, pruebas de todo |
| 2 | Cinéfilos | 373 | 170 | 3.75 | Maratonianos, muchas películas, buen rating |

##  Tecnologías

- **Python 3.8+**
- **scikit-learn**: K-Means clustering
- **pandas**: Manipulación de datos
- **Streamlit**: Interfaz web
- **Matplotlib/Seaborn**: Visualizaciones
- **joblib**: Persistencia de modelos

##  Dataset

[MovieLens 100K](https://grouplens.org/datasets/movielens/100k/)
- 943 usuarios
- 1,682 películas
- 100,000 ratings
- Años: 1997-1998

##  Notas

- El modelo K-Means se entrena 1 sola vez en `main.py`
- Se persiste con `joblib` para carga rápida
- La app usa búsqueda de usuarios similares para mayor personalización
- Las recomendaciones son dinámicas según edad, película y ratings

## 👨 Autor

Israel Rodriguez Gonzalez

**Para comenzar:** `streamlit run PYTHON/app_streamlit.py`

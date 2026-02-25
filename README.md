#  Movie Recommendation System

Intelligent movie recommendation system based on K-Means clustering using the MovieLens 100K dataset. It groups users into 3 profiles and generates personalized recommendations.

##  Features

-  **K-Means Clustering**: Segments 943 users into 3 groups
-  **25 Features per user**: Age, average rating, movies watched, genre preferences
-  **Personalized recommendations**: Based on similarity with users from the same cluster
-  **Streamlit Interface**: Interactive and user-friendly app
-  **5 Cluster** analysis visualizations

##  Requirements

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

Install all dependencies:
```bash
pip install -r requirements.txt
```

##  3 Clusters

| Cluster | Name | Users | Avg. Movies | Avg. Rating | Profile |
|---------|--------|----------|--------------|--------------|--------|
| 0 | Selective Users | 312 | 43 | 3.81 | Watch few movies but give high ratings |
| 1 | Active Critics | 258 | 90 | 3.08 | Very active, try everything |
| 2 | Cinephiles | 373 | 170 | 3.75 | Binge-watchers, many movies, good ratings |

##  Technologies

- **Python**
- **scikit-learn**: K-Means clustering
- **pandas**: Data manipulation
- **Streamlit**: Web interface
- **Matplotlib/Seaborn**: Visualizations
- **joblib**: Model persistence

##  Dataset

[MovieLens 100K](https://grouplens.org/datasets/movielens/100k/)
- 943 users
- 1,682 movies
- 100,000 ratings
- Years: 1997-1998

##  Notes

- The K-Means model is trained only once in `main.py`
- The trained model is persisted using `joblib` for fast loading
- The app uses similar user search for improved personalization
- Recommendations are dynamic based on age, movies, and ratings

## Autor

isra19dev

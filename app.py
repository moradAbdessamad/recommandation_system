from flask import Flask, render_template, request, redirect, url_for, jsonify #type: ignore
import pandas as pd
import os
import numpy as np 
import sqlite3
import faiss #type: ignore
import pickle
import requests
import tensorflow as tf #type: ignore
from tensorflow.keras.applications import ResNet50 #type: ignore
from tensorflow.keras.preprocessing import image #type: ignore
from tensorflow.keras.applications.resnet50 import preprocess_input #type: ignore
from tensorflow.keras.models import Model #type: ignore
from typing import List, Dict
from datetime import datetime 
from flask_cors import CORS #type: ignore
from werkzeug.utils import secure_filename #type: ignore


app = Flask(__name__)
CORS(app)

csv_path = '../Data/movies_data_with_textual_representation.csv'
db_name = 'SQLDB/movies_database.db'
table_name = 'movies_data'
save_text_faiss_path = "../Models/faiss_index"
dim = 4096
RECOMMENDATIONS_TABLE = 'recommendations_history'
save_dir = '../Models'
UPLOAD_FOLDER = 'temp_uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# remove the final FC layer to get a 2048-d output
base_model = ResNet50(weights='imagenet')
model = Model(
    inputs=base_model.input,
    outputs=base_model.get_layer('avg_pool').output
)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_textual_representation(row):
    """
    Combine various textual fields into a single textual representation.
    """
    textual_representation = f"""
    Types : {row['genres']}
    Title : {row['title']}
    Director : {row['director']}
    Cast : {row['actors']}
    Realeased : {row['release_date']}
    Country : {row['production_countries']}
    Language : {row['original_language']}
    Description : {row['overview']}
    """
    return textual_representation


def initialize_recommender():
    """
    Initialize the recommendation system by loading the FAISS index and movie data.
    Also creates necessary database tables.
    """
    print("Initializing recommendation system...")
    global index, df
    index = faiss.read_index(save_text_faiss_path)
    df = pd.read_csv(csv_path)
    create_recommendations_table()
    create_poster_recommendations_table()
    print("Recommendation system initialized successfully")


def make_text_embedding(movie_text: str) -> np.ndarray:
    """
    Create embedding for movie text using the local Ollama API
    """
    response = requests.post(
        'http://localhost:11434/api/embeddings',
        json={
            'model': 'llama2',
            'prompt': movie_text,
            "device": "cuda",
        }
    )
    
    if response.status_code != 200:
        raise Exception(f"Failed to get embedding: {response.text}")
    
    embedding = np.array(response.json()['embedding'], dtype=np.float32)
    return embedding


def get_movie_recommendations(text_embedding: np.ndarray, num_results: int = 6) -> List[Dict]:
    """
    Get movie recommendations based on text embedding with full movie details
    """
    distances, indices = index.search(text_embedding.reshape(1, -1), num_results)
    
    recommendations = []
    for i, idx in enumerate(indices[0]):
        movie_data = df.iloc[idx].to_dict()
        
        # Ensure numeric values are properly handled
        movie_details = {
            'similarity_score': float(distances[0][i]),
            'title': str(movie_data.get('title', '')),
            'genres': str(movie_data.get('genres', '')),
            'director': str(movie_data.get('director', '')),
            'actors': str(movie_data.get('actors', '')),
            'release_date': str(movie_data.get('release_date', '')),
            'production_countries': str(movie_data.get('production_countries', '')),
            'original_language': str(movie_data.get('original_language', '')),
            'overview': str(movie_data.get('overview', '')),
            'vote_average': float(movie_data.get('vote_average', 0)),
            'vote_count': int(movie_data.get('vote_count', 0)),
            'popularity': float(movie_data.get('popularity', 0)),
            'runtime': int(movie_data.get('runtime', 0)),
            'budget': int(movie_data.get('budget', 0)),
            'revenue': int(movie_data.get('revenue', 0)),
            'tagline': str(movie_data.get('tagline', '')),
            'poster_path': str(movie_data.get('poster_path', '')),
            'textual_representation': str(movie_data.get('Textual_representation', ''))
        }
        recommendations.append(movie_details)
    
    return recommendations


def create_recommendations_table():
    """
    Create the recommendations history table if it doesn't exist
    """
    create_table_query = """
    CREATE TABLE IF NOT EXISTS recommendations_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME,
        input_text TEXT,
        recommended_movie_id INTEGER,
        similarity_score REAL,
        FOREIGN KEY (recommended_movie_id) REFERENCES movies_data (id)
    );
    """
    try:
        with sqlite3.connect(db_name) as conn:
            conn.execute(create_table_query)
    except Exception as e:
        print(f"Error creating recommendations table: {e}")
        raise


def save_recommendations(input_text: str, recommendations: List[Dict]):
    """
    Save recommendations to the database
    """
    try:
        with sqlite3.connect(db_name) as conn:
            cursor = conn.cursor()
            timestamp = datetime.now().isoformat()
            
            for rec in recommendations:
                # First try to find the movie by title and release date
                cursor.execute(
                    f"""
                    SELECT id 
                    FROM {table_name} 
                    WHERE title = ? 
                    AND release_date = ?
                    LIMIT 1
                    """,
                    (rec['title'], rec['release_date'])
                )
                
                result = cursor.fetchone()
                
                if result:
                    movie_id = result[0]
                    
                    # Check if this recommendation already exists
                    cursor.execute(
                        f"""
                        SELECT id FROM {RECOMMENDATIONS_TABLE}
                        WHERE timestamp = ? 
                        AND recommended_movie_id = ?
                        """,
                        (timestamp, movie_id)
                    )
                    
                    if not cursor.fetchone():  
                        cursor.execute(
                            f"""
                            INSERT INTO {RECOMMENDATIONS_TABLE} 
                            (timestamp, input_text, recommended_movie_id, similarity_score)
                            VALUES (?, ?, ?, ?)
                            """,
                            (timestamp, input_text, movie_id, rec['similarity_score'])
                        )
                else:
                    print(f"Warning: Movie not found in database: {rec['title']}")
            
            conn.commit()
            
    except sqlite3.Error as e:
        print(f"SQLite error saving recommendations: {e}")
        raise
    
    except Exception as e:
        print(f"Error saving recommendations: {e}")
        raise


def load_faiss_index(save_dir='../Models'):
    """
    Loads the previously saved FAISS index and movie IDs.
    """
    index_path = os.path.join(save_dir, 'poster_features.index')
    movie_ids_path = os.path.join(save_dir, 'movie_ids.pkl')

    index = faiss.read_index(index_path)
    with open(movie_ids_path, 'rb') as f:
        movie_ids = pickle.load(f)

    return index, movie_ids


def get_image_features(img_path):
    """
    Loads an image, preprocesses it, 
    and returns the flattened 2048-d ResNet50 feature vector.
    """
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)        
    img_array = preprocess_input(img_array)              
    features = model.predict(img_array)                 
    return features.flatten()     


def find_similar_movies(image_file, index, movie_ids, k=6):
    """
    Given an uploaded image file and a FAISS index, 
    returns the top-K closest movie IDs based on L2 distance.
    """
    try:
        # Get features directly from the image file
        img = image.load_img(image_file, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        query_features = model.predict(img_array)
        
        query_features = query_features.astype('float32')
        query_features = np.reshape(query_features, (1, -1))  # Ensure correct shape

        distances, indices = index.search(query_features, k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            movie_id = movie_ids[idx]
            similarity = 1 / (1 + dist)
            results.append((movie_id, similarity))

        return results
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        raise

def create_poster_recommendations_table():
    """
    Create the poster recommendations history table if it doesn't exist
    """
    create_table_query = """
    CREATE TABLE IF NOT EXISTS poster_recommendations_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME,
        image_path TEXT,
        recommended_movie_id INTEGER,
        similarity_score REAL,
        FOREIGN KEY (recommended_movie_id) REFERENCES movies_data (id)
    );
    """
    try:
        with sqlite3.connect(db_name) as conn:
            conn.execute(create_table_query)
    except Exception as e:
        print(f"Error creating poster recommendations table: {e}")
        raise

def save_poster_recommendations(image_path: str, recommendations: List[Dict]):
    """
    Save poster recommendations to the database
    """
    try:
        with sqlite3.connect(db_name) as conn:
            cursor = conn.cursor()
            timestamp = datetime.now().isoformat()
            
            for rec in recommendations:
                movie_id = rec['id']
                cursor.execute(
                    """
                    INSERT INTO poster_recommendations_history 
                    (timestamp, image_path, recommended_movie_id, similarity_score)
                    VALUES (?, ?, ?, ?)
                    """,
                    (timestamp, image_path, movie_id, rec['similarity_score'])
                )
            conn.commit()
    except Exception as e:
        print(f"Error saving poster recommendations: {e}")
        raise


@app.route('/recommande-poster-movie', methods=['POST'])
def recommande_movie_poster_method():
    """
    Process an uploaded movie poster image and return similar movies.
    JSON response containing movie recommendations based on poster similarity
    """
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed. Use PNG, JPG, or JPEG'}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        try:
            index, movie_ids = load_faiss_index()
            similar_movie = find_similar_movies(filepath, index, movie_ids)

            recommandations_poster = []
            for movie_id, similarity in similar_movie:
                with sqlite3.connect(db_name) as conn:
                    cursor = conn.cursor()
                    cursor.execute(f"SELECT * FROM {table_name} WHERE id = ?", (movie_id,))
                    movie_data = cursor.fetchone()

                    if movie_data:
                        columns = [description[0] for description in cursor.description]
                        movie_dict = dict(zip(columns, movie_data))
                        movie_dict['similarity_score'] = float(similarity)
                        recommandations_poster.append(movie_dict)

            try:
                save_poster_recommendations(filename, recommandations_poster)
            except Exception as e:
                print(f"Warning: Failed to save poster recommendations: {e}")

            return jsonify({
                'status': 'success',
                'movie_count': len(recommandations_poster),
                'recommendations': recommandations_poster
            })

        finally:
            try:
                os.remove(filepath)
            except Exception as e:
                print(f"Warning: Failed to remove temporary file: {e}")

    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@app.route('/recommande-text-movie', methods=['POST'])
def recommande_text_movie_method():
    """
    Process a text description and return similar movies.
    JSON response containing movie recommendations based on text similarity
    """
    try:
        data = request.get_json()
        if not data or 'movie_text' not in data:
            return jsonify({'error': 'No movie_text provided in request'}), 400
        
        movie_text = data['movie_text']
        num_results = data.get('num_results', 6) 
        
        try:
            embedding = make_text_embedding(movie_text)
        except Exception as e:
            return jsonify({'error': f'Failed to generate embedding: {str(e)}'}), 500
        
        recommendations = get_movie_recommendations(embedding, num_results)

        try:
            save_recommendations(movie_text, recommendations)
        except Exception as e:
            print(f"Warings: faild to save recommandations: {e}")            
        
        return jsonify({
            'movie_count': len(recommendations),
            'status': 'success',
            'recommendations': recommendations
        })
    
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@app.route('/recommendation-poster-history', methods=['GET'])
def get_recommendation_poster_history():
    """
    Get the history of poster-based recommendations
    """
    try:
        with sqlite3.connect(db_name) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query = """
            SELECT 
                ph.timestamp,
                ph.image_path,
                ph.similarity_score,
                m.*
            FROM poster_recommendations_history ph
            JOIN movies_data m ON ph.recommended_movie_id = m.id
            ORDER BY ph.timestamp DESC
            LIMIT 100
            """
            
            cursor.execute(query)
            history = [dict(row) for row in cursor.fetchall()]
            
            return jsonify({
                'status': 'success',
                'history': history
            })
            
    except Exception as e:
        return jsonify({'error': f'Failed to fetch poster history: {str(e)}'}), 500

@app.route('/recommendation-text-history', methods=['GET'])
def get_recommendation_text_history():
    """
    Get the history of text-based recommendations. Returns a JSON response containing
    the most recent 100 text-based movie recommendations with their details.
    """
    try:
        with sqlite3.connect(db_name) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query = """
            SELECT 
                rh.timestamp,
                rh.input_text,
                rh.similarity_score,
                m.*
            FROM recommendations_history rh
            JOIN movies_data m ON rh.recommended_movie_id = m.id
            ORDER BY rh.timestamp DESC
            LIMIT 100
            """
            
            cursor.execute(query)
            history = [dict(row) for row in cursor.fetchall()]
            
            return jsonify({
                'status': 'success',
                'history': history
            })
            
    except Exception as e:
        return jsonify({'error': f'Failed to fetch history: {str(e)}'}), 500



@app.route('/upload-movies', methods=['POST'])
def upload_movies():
    """
    Upload movies data from CSV to SQLite database.
    Creates a movies table if it doesn't exist and populates it with data from the CSV file.
    Returns JSON response indicating success or failure.
    """
    # SQL table creation query
    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        id INTEGER PRIMARY KEY,
        title TEXT,
        original_title TEXT,
        release_date TEXT,
        overview TEXT,
        genres TEXT,
        keywords TEXT,
        runtime REAL,
        vote_average REAL,
        vote_count INTEGER,
        popularity REAL,
        budget INTEGER,
        revenue INTEGER,
        original_language TEXT,
        poster_path TEXT,
        backdrop_path TEXT,
        director TEXT,
        actors TEXT,
        production_companies TEXT,
        production_countries TEXT,
        Textual_representation TEXT
    );
    """

    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        
        cursor.execute(create_table_query)
        
        df = pd.read_csv(csv_path)
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        
        conn.commit()
        return jsonify({'message': 'Movies data uploaded successfully'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        if conn:
            conn.close()


@app.route('/get-movies')
def get_all_movies():
    """
    Retrieve all movies from the database and return as JSON
    200: JSON with movie count and list of all movies
    404: If no movies found
    500: Server error
    """
    try:
        conn = sqlite3.connect(db_name)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = f"SELECT * FROM {table_name}"
        cursor.execute(query)
        
        movies_data = cursor.fetchall()
        if not movies_data:
            return jsonify({'message': 'No movies found'}), 404
        
        movies_list = [dict(movie) for movie in movies_data]

        return jsonify({
            'count': len(movies_list),
            'movies': movies_list,
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    except sqlite3.Error as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if 'conn' in locals():
            conn.close()


@app.route('/get-movie/<int:movie_id>')
def get_specific_movie(movie_id):
    """
    Retrieve a specific movie from the database based on its ID.
    """
    try:
        conn = sqlite3.connect(db_name)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = f"SELECT * FROM {table_name} WHERE id = ?"
        cursor.execute(query, (movie_id,))
        
        movie_data = cursor.fetchone()
        if not movie_data:
            return jsonify({'message': 'Movie not found'}), 404
        
        movie_dict = dict(movie_data)

        return jsonify(movie_dict), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    except sqlite3.Error as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    initialize_recommender()
    app.run(debug=True)
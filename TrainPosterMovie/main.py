import os
import pickle

import numpy as np
import faiss #type: ignore

import tensorflow as tf #type: ignore
from tensorflow.keras.applications import ResNet50 #type: ignore
from tensorflow.keras.preprocessing import image #type: ignore
from tensorflow.keras.applications.resnet50 import preprocess_input #type: ignore
from tensorflow.keras.models import Model #type: ignore


# remove the final FC layer to get a 2048-d output
base_model = ResNet50(weights='imagenet')
model = Model(
    inputs=base_model.input,
    outputs=base_model.get_layer('avg_pool').output
)


def get_image_features(img_path):
    """
    Loads an image, preprocesses it, 
    and returns the flattened 2048-d ResNet50 feature vector.
    """
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)        # shape: (1, 224, 224, 3)
    img_array = preprocess_input(img_array)              # ResNet50 preprocessing
    features = model.predict(img_array)                  # shape: (1, 2048)
    return features.flatten()                            # shape: (2048,)

def build_and_save_faiss_index(poster_dir, save_dir='../Models'):
    """
    Iterates over images in poster_dir, extracts features, 
    creates a FAISS L2 index,
    and saves the index + the corresponding movie IDs.
    """
    os.makedirs(save_dir, exist_ok=True)

    features_list = []
    movie_ids = []

    print("Extracting features from images...")
    for filename in os.listdir(poster_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            movie_id = os.path.splitext(filename)[0]  
            image_path = os.path.join(poster_dir, filename)

            try:
                features = get_image_features(image_path)
                features_list.append(features)
                movie_ids.append(movie_id)
                print(f"Processed: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    features_array = np.array(features_list, dtype='float32')
    print(f"Feature array shape: {features_array.shape}")

    dimension = features_array.shape[1]
    index = faiss.IndexFlatL2(dimension)   # L2 distance index
    index.add(features_array)

    index_path = os.path.join(save_dir, 'poster_features.index')
    faiss.write_index(index, index_path)
    print(f"FAISS index saved to {index_path}")

    movie_ids_path = os.path.join(save_dir, 'movie_ids.pkl')
    with open(movie_ids_path, 'wb') as f:
        pickle.dump(movie_ids, f)
    print(f"Movie IDs saved to {movie_ids_path}")

    return index, movie_ids


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


def find_similar_movies(query_image_path, index, movie_ids, k=5):
    """
    Given a query image and a FAISS index, 
    returns the top-K closest movie IDs based on L2 distance.
    """
    query_features = get_image_features(query_image_path).astype('float32')
    query_features = np.expand_dims(query_features, axis=0)  # shape: (1, 2048)

    distances, indices = index.search(query_features, k)  

    results = []
    for idx, dist in zip(indices[0], distances[0]):
        movie_id = movie_ids[idx]
        similarity = 1 / (1 + dist)
        results.append((movie_id, similarity))

    return results


if __name__ == "__main__":
    index, movie_ids = load_faiss_index("../Models")
    similar_movie = find_similar_movies("../Data/images/5.jpg", index, movie_ids)

    for movie_id, similarity in similar_movie:
        print(f"Movie ID: {movie_id}, Similarity: {similarity:.3f}")

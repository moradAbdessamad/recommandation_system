import faiss #type: ignore
import numpy as np #type: ignore
import requests #type: ignore
import os
import pandas as pd #type: ignore

save_path = "../Models/faiss_index"
dim = 4096

Text_movie = """
    Types : Action, Science Fiction, Adventure
    Title : nan
    Director : Martin Scorsese,
    Cast : leonardo dicaprio, joseph gordon-levitt
    Realeased : 2010-07-15
    Country : United Kingdom, United States of America
    Language : en
    Description : a group of the world's most dangerous criminals are brought together to form a team of the world's most dangerous criminals. they are tasked with the impossible - to plant an idea in someone's mind and make them believe it is their own. the team must navigate through a world of dreams and reality to complete their mission.
"""

def create_textual_representation(row):
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

def make_text_emebedding(movie_text):
    print("Creating embedding for the movie text...")

    res = requests.post(
        'http://localhost:11434/api/embeddings',
        json={
            'model': 'llama2',
            'prompt': movie_text,
            "device": "cuda",
        }
    )

    embedding_movie_text = np.array(res.json()['embedding'], dtype=np.float32)

    if res.status_code != 200:
        raise Exception(f"Failed to get embedding: {res.text}")
    else:
        print("Embedding created successfully")

    return embedding_movie_text


def recommande_text_movie(text_embedding, num_results=5):
    print("Loading the faiss index and DF...")
    index = faiss.read_index(save_path)
    df = pd.read_csv('../Data/movies_data_with_textual_representation.csv')

    print("Searching for the best matches...")
    D, I = index.search(text_embedding.reshape(1, -1), num_results)
    best_matches = np.array(df['Textual_representation'])[I.flatten()]

    print("The best matches are:")
    for match in best_matches:
        print("-----------------------------")
        print(match)

    return I


if __name__ == "__main__":
    embedding_movie_text = make_text_emebedding(Text_movie)
    recommande_text_movie(embedding_movie_text)


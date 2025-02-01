import requests #type: ignore
import pandas as pd #type: ignore
import json 
from time import sleep
import os

based_url = 'https://api.themoviedb.org/3'

api_key = '984a44958b896d4b6d728a3c07bb63c2'

headers = {
    "accept": "application/json",
    "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiI5ODRhNDQ5NThiODk2ZDRiNmQ3MjhhM2MwN2JiNjNjMiIsIm5iZiI6MTczNTY1MzE4NC4wMjQsInN1YiI6IjY3NzNmNzQwM2ZjNzZlYTU4ODkyYTY1ZSIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.8OA82C4eVt2wMS0RbyFHGY5-ZbAqRK55XPmSCM7g51g"
}

def write_to_csv(movies_data, file_path, is_first_page=False):
    """
    Write movie data to a CSV file.

    Args:
        movies_data: List of dictionaries containing movie information
        file_path: Path to the CSV file
        is_first_page: Boolean indicating if this is the first page of data
    """
    df = pd.DataFrame(movies_data)
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    
    if is_first_page:
        df.to_csv(file_path, index=False, encoding='utf-8', mode='w')
    else:
        df.to_csv(file_path, index=False, encoding='utf-8', mode='a', header=False)


def get_movies_tmdb(api_key, file_path, start_page=1, end_page=10):
    """
    Fetch movie data from TMDB API and save to CSV file.

    Args:
        api_key: TMDB API key
        file_path: Path to save CSV file
        start_page: Starting page number for API requests
        end_page: Ending page number for API requests
    """
    for page in range(start_page, end_page+1):
        try: 
            movie_url = f"{based_url}/discover/movie?include_adult=false&include_video=false&language=en-US&page={page}&sort_by=popularity.desc"
            
            reponse = requests.get(movie_url, headers=headers)
            movies = reponse.json()['results']
            
            page_movies_data = []
            
            for movie in movies:
                movie_id = movie['id']
                
                details_url = f"{based_url}/movie/{movie_id}?api_key={api_key}"
                details_url_reponse = requests.get(details_url, headers=headers)
                
                movie_details = details_url_reponse.json()
                
                movie_data = {
                    "id": movie_id,
                    "title": movie_details.get("title", ""),
                    "original_title": movie_details.get("original_title", ""),
                    "release_date": movie_details.get("release_date", ""),
                    "overview": movie_details.get("overview", ""),
                    "genres": ", ".join([genre["name"] for genre in movie_details.get("genres", [])]),
                    "keywords": ", ".join([keyword["name"] for keyword in movie_details.get("keywords", {}).get("keywords", [])]),
                    "runtime": movie_details.get("runtime", 0),
                    "vote_average": movie_details.get("vote_average", 0),
                    "vote_count": movie_details.get("vote_count", 0),
                    "popularity": movie_details.get("popularity", 0),
                    "budget": movie_details.get("budget", 0),
                    "revenue": movie_details.get("revenue", 0),
                    "original_language": movie_details.get("original_language", ""),
                    "poster_path": movie_details.get("poster_path", ""),
                    "backdrop_path": movie_details.get("backdrop_path", ""),
                    "director": ", ".join([crew["name"] for crew in movie_details.get("credits", {}).get("crew", []) if crew["job"] == "Director"]),
                    "actors": ", ".join([cast["name"] for cast in movie_details.get("credits", {}).get("cast", [])[:5]]),
                    "production_companies": ", ".join([company["name"] for company in movie_details.get("production_companies", [])]),
                    "production_countries": ", ".join([country["name"] for country in movie_details.get("production_countries", [])])
                }
                page_movies_data.append(movie_data)
            
            write_to_csv(page_movies_data, file_path, is_first_page=(page==start_page))
            
            print(f"Completed Page: {page}/{end_page}")
            sleep(0.25)

        except Exception as e:
            print(f"Error: {e} in the Page {page}")
            continue


def main():
    data_file = '../Data/movies_data.csv'
    
    os.makedirs(os.path.dirname(data_file), exist_ok=True)
    
    get_movies_tmdb(api_key, data_file, start_page=1, end_page=500)
    
    print(f"Data collection completed: {data_file}")


if __name__ == "__main__":
    main()
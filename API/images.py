import pandas as pd  # type: ignore
import requests  # type: ignore
import os

def download_posters(csv_path, save_dir):
    """
    Download movie posters from TMDB API using poster paths from a CSV file.
        
    Parameters:
        csv_path (str): Path to CSV file containing movie data with poster_path and id columns
        save_dir (str): Directory path where poster images will be saved
        
    Returns:
        None
    """
    df = pd.read_csv(csv_path)
    df = df.sample(5000)

    base_url = 'https://image.tmdb.org/t/p/w500'
    os.makedirs(save_dir, exist_ok=True)
    
    for _, row in df.iterrows():
        if pd.notna(row['poster_path']):
            poster_path = row['poster_path'].strip('/')
            movie_id = row['id']
            save_path = os.path.join(save_dir, f"{movie_id}.jpg")
            
            # Skip downloading if the file already exists
            if os.path.exists(save_path):
                print(f"Skipped: {movie_id}.jpg (already exists)")
                continue
            
            poster_url = f"{base_url}/{poster_path}"
            
            try:
                response = requests.get(poster_url)
                response.raise_for_status()
                
                # Save the poster
                with open(save_path, 'wb') as f:
                    f.write(response.content)
                print(f"Downloaded: {movie_id}.jpg")
                
            except Exception as e:
                print(f"Error downloading {movie_id}: {e}")

if __name__ == "__main__":
    csv_path = '../Data/movies_data.csv'
    save_dir = '../Data/images'
    download_posters(csv_path, save_dir)

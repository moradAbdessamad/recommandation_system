import { useState } from "react"
import MovieCard from "./MovieCard"
import { fetchTextRecommendations } from "../APIs/recommendationText"

const TextSection = () => {
  const [preferences, setPreferences] = useState("");
  const [recommendations, setRecommendations] = useState([]); 
  const [isLoading, setIsLoading] = useState(false);

  const handleGetRecommendations = async () => {  
    try {
      setIsLoading(true);
      const data = await fetchTextRecommendations(preferences, 6);
      setRecommendations(data.recommendations || []);
    } catch (error) {
      console.error("Failed to fetch text recommendations:", error);
    } finally {
      setIsLoading(false);
    }
  };


  return (
    <div className="recommendation-section">
      
      <h2>Text-based Recommendation</h2>
      
      <div className="input-group">
        <label htmlFor="movie-preferences">Enter your movie preferences:</label>
        <textarea
          id="movie-preferences"
          rows={4}
          placeholder="E.g., I like sci-fi movies with strong female leads..."
          value={preferences}
          onChange={(e) => setPreferences(e.target.value)}
        ></textarea>
      </div>
      
      <button onClick={handleGetRecommendations}>Get Text Recommendations</button>
      
{isLoading && (
        <div className="results">
          <p>Recommending…</p>
        </div>
      )}

      {!isLoading && recommendations.length > 0 && (
        <div className="results">
          {recommendations.map((movie, index) => {
            const {
              poster_path,
              title,
              vote_average,
              runtime,
              release_date,
            } = movie;

            return (
              <MovieCard
                key={index}
                title={title}
                poster={`https://image.tmdb.org/t/p/w500${poster_path}`}
                voteAverage={vote_average}
                runningTime={`${runtime} min`}
                lastUpdate={release_date} 
              />
            );
          })}
        </div>
      )}

    </div>
  )
}

export default TextSection


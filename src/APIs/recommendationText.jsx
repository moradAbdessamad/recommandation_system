export const fetchTextRecommendations = async (movieText, numResults = 6) => {
    if (!movieText?.trim()) {
      throw new Error('Movie text is required');
    }
  
    try {
      const response = await fetch('http://localhost:5000/recommande-text-movie', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify({
          movie_text: movieText,
          num_results: numResults
        })
      });
  
      if (!response.ok) {
        const errorData = await response.json().catch(() => null);
        throw new Error(
          errorData?.error || 
          `Server responded with ${response.status}: ${response.statusText}`
        );
      }
  
      const data = await response.json();
  
      if (!data.recommendations || !Array.isArray(data.recommendations)) {
        throw new Error('Invalid response format from server');
      }
  
      return {
        movie_count: data.movie_count || 0,
        status: data.status || 'error',
        recommendations: data.recommendations.map(movie => ({
          title: String(movie.title || ''),
          poster_path: String(movie.poster_path || ''),
          vote_average: Number.isFinite(movie.vote_average) ? movie.vote_average : 0,
          runtime: Number.isFinite(movie.runtime) ? movie.runtime : 0,
          release_date: String(movie.release_date || ''),
          overview: String(movie.overview || ''),
          genres: String(movie.genres || ''),
          director: String(movie.director || ''),
          actors: String(movie.actors || ''),
          similarity_score: Number.isFinite(movie.similarity_score) ? movie.similarity_score : 0
        }))
      };
  
    } catch (error) {
      console.error('Error fetching text recommendations:', error);
      throw new Error(error.message || 'Failed to fetch recommendations');
    }
  };
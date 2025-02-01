export const fetchTextRecommendationHistory = async () => {
    try {
      const response = await fetch('http://localhost:5000/recommendation-text-history');
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      return data.history || [];
    } catch (error) {
      console.error('Error fetching text recommendation history:', error);
      throw error;
    }
  };
  
  export const fetchPosterRecommendationHistory = async () => {
    try {
      const response = await fetch('http://localhost:5000/recommendation-poster-history');
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      return data.history || [];
    } catch (error) {
      console.error('Error fetching poster recommendation history:', error);
      throw error;
    }
  };
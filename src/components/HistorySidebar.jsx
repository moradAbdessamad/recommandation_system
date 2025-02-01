// HistorySidebar.js
import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom'; 
import {
  fetchPosterRecommendationHistory,
  fetchTextRecommendationHistory
} from '../APIs/recommendationHistory';

const HistorySidebar = ({ isOpen }) => {
  const [textHistory, setTextHistory] = useState([]);
  const [posterHistory, setPosterHistory] = useState([]);

  useEffect(() => {
    const getHistoryData = async () => {
      try {
        const textData = await fetchTextRecommendationHistory();
        setTextHistory(textData);
      } catch (error) {
        console.error('Error fetching text recommendation history:', error);
      }

      try {
        const posterData = await fetchPosterRecommendationHistory();
        setPosterHistory(posterData);
      } catch (error) {
        console.error('Error fetching poster recommendation history:', error);
      }
    };
    getHistoryData();
  }, []);

  return (
    <div className={`history-sidebar ${isOpen ? "" : "closed"}`}>

      <div className="history-section text-history">
        <h3>History Text Recommandation</h3>
        
        <div id="recent-history-text-container">
          {textHistory.map((item, idx) => (
            <div className='history-poster' key={idx}>
              <Link to={`/movie/${item.id}`} style={{ textDecoration: 'none', color: 'inherit' }}>
                {item.input_text}
              </Link>
            </div>
          ))}
        </div>

      </div>

      <div className="history-section poster-history">
        <h3>History Poster Recommandation</h3>

        <div id="recent-history-poster-container">
          {posterHistory.map((item, idx) => (
            <div className='history-poster' key={idx}>
              <Link to={`/movie/${item.id}`} style={{ textDecoration: 'none', color: 'inherit' }}>
                {item.overview}
              </Link>
            </div>
          ))}
        </div>
        
      </div>

    </div>
  );
};

export default HistorySidebar;

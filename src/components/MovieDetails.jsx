import React, { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import SingleMovieCard from "../components/SingleMovieCard";

const MovieDetails = () => {
  const { id } = useParams();
  const [movie, setMovie] = useState(null);

  useEffect(() => {
    fetch(`http://localhost:5000/get-movie/${id}`)
      .then((res) => res.json())
      .then((data) => setMovie(data))
      .catch((error) => console.error("Error fetching movie details:", error));
  }, [id]);

  if (!movie) {
    return <div>Loading movie details...</div>;
  }

  return (
    <div className="movie-details-container">
      <div className="movie-details-header">
        <a href="/" className="home-link">
          <h2>Text-based Recommendation</h2>
        </a>
      </div>

      <div className="movie-details-card">
        <SingleMovieCard
          title={movie.title || "Unknown Title"}
          year={movie.release_date || "Unknown Year"}
          director={movie.director || "Unknown Director"}
          duration={movie.runtime ? `${movie.runtime} min` : "N/A"}
          genres={movie.genres ? movie.genres.split(",").map((genre) => genre.trim()) : ["Unknown Genre"]}
          description={movie.overview || "No description available."}
          posterUrl={movie.poster_path ? `https://image.tmdb.org/t/p/w500${movie.poster_path}` : "../Data/100.jpg"}
        />
      </div>
    </div>
  );
};

export default MovieDetails;

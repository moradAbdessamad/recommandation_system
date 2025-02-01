import { useState } from "react";
import MovieCard from "./MovieCard";

const ImageSection = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [recommendations, setRecommendations] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [previewUrl, setPreviewUrl] = useState(null);

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      // Create a preview URL for the selected image
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
    }
  };

  const getImageRecommendation = async () => {
    if (!selectedFile) {
      alert("Please select an image first");
      return;
    }

    setIsLoading(true);
    const formData = new FormData();
    formData.append("image", selectedFile);

    try {
      const response = await fetch(
        "http://localhost:5000/recommande-poster-movie",
        {
          method: "POST",
          body: formData,
        }
      );

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setRecommendations(data.recommendations || []);
    } catch (error) {
      console.error("Error uploading image:", error);
      alert("Failed to get recommendations");
      setRecommendations(null);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="recommendation-section">
      <h2>Image-based Recommendation</h2>
      <div className="image-upload">
        <label htmlFor="image-upload">
          Upload Image
          <input
            type="file"
            id="image-upload"
            accept="image/png, image/jpeg"
            onChange={handleFileSelect}
          />
        </label>
      </div>

      {previewUrl && (
        <div className="preview-container">
          <img
            src={previewUrl}
            alt="Preview"
            className="preview-image"
            style={{ maxWidth: "200px", marginTop: "1rem" }}
          />
        </div>
      )}

      <button
        onClick={getImageRecommendation}
        disabled={!selectedFile || isLoading}
      >
        {isLoading ? "Getting Recommendations..." : "Get Image Recommendations"}
      </button>

      {isLoading && (
        <div className="results">
          <p>Processing your image...</p>
        </div>
      )}

      {!isLoading && recommendations && (
        <div className="results">
          {recommendations.map((movie, index) => (
            <MovieCard
              key={index}
              title={movie.title}
              poster={`https://image.tmdb.org/t/p/w500${movie.poster_path}`}
              voteAverage={movie.vote_average}
              runningTime={`${movie.runtime} min`}
              lastUpdate={movie.release_date}
            />
          ))}
        </div>
      )}
    </div>
  );
};

export default ImageSection;

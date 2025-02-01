const SingleMovieCard = ({ title, year, director, duration, genres, description, posterUrl }) => {
    return (
      <div className="movie-card-single">
        <div className="left-side-single">
          <img
            src={posterUrl ? `https://image.tmdb.org/t/p/w500${posterUrl}` : "/placeholder.svg"}
            alt={title}
            className="poster-single"
          />
        </div>
        <div className="right-side-single">
          <div
            className="background-image-single"
            style={{
              backgroundImage: `url(${posterUrl ? `https://image.tmdb.org/t/p/w500${posterUrl}` : "/placeholder.svg"})`
            }}
          ></div>
          <h2 className="title-single">{title}</h2>
          <p className="subtitle-single">
            {year} • {director}
          </p>
          <span className="duration-single">{duration}</span>
          <div className="genres-single">
            {genres?.map((genre, index) => (
              <span key={index} className="genre-single">
                {genre}
              </span>
            ))}
          </div>
          <p className="description">{description}</p>
          <div className="glow-effect"></div>
        </div>
      </div>
    );
  };
  
  export default SingleMovieCard;
  